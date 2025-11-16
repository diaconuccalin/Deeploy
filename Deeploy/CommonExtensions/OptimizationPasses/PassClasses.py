# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Optimization Pass Framework - Composable Graph Transformations

This module implements the foundation for all optimization passes in Deeploy,
providing a composable, reusable framework for transforming ONNX computation graphs.

Pass Architecture:
    Deeploy uses a pass-based compiler architecture where graph transformations
    are implemented as modular "passes" that can be:
    - Composed into sequences (SequentialPass)
    - Reused across different platforms
    - Conditionally applied based on graph structure
    - Either context-aware (access to NetworkContext) or context-agnostic

Key Concepts:
    1. **Pass**: Base class for all optimization passes. Can contain subpasses
       for hierarchical composition.

    2. **Context-Aware vs Context-Agnostic**:
       - Context-aware: Has access to NetworkContext (buffers, memory, type info)
       - Context-agnostic: Works only on the graph structure
       - Use @contextaware or @contextagnostic decorators to add the appropriate MixIn

    3. **Retargeting**: Some passes are graph-specific (e.g., pattern replacement).
       The retarget() method allows adapting a pass to a new graph instance.

    4. **Pattern Replacement**: ReplaceSequentialPatternPass finds all instances
       of a pattern in the graph and replaces them with new nodes.

Pass Types:
    - Pass: Base class for all passes
    - SequentialPass: Executes a sequence of subpasses in order
    - ReplaceMatchWithModulePass: Replaces a single matched pattern
    - ReplaceSequentialPatternPass: Finds and replaces all pattern instances

Example:
    # Define a simple context-agnostic pass
    @contextagnostic
    class MyOptimizationPass(Pass):
        def run_pass(self, graph: gs.Graph) -> gs.Graph:
            # Transform graph
            return graph

    # Create a sequential pass combining multiple optimizations
    optimizer = SequentialPass(
        TransposeOptimizationPass(),
        ConstantFoldingPass(),
        DeadCodeEliminationPass()
    )

    # Apply to graph
    optimized_graph = optimizer(graph)

Usage with Decorators:
    The @contextaware and @contextagnostic decorators automatically add the
    appropriate MixIn class to handle pass application, retargeting, and
    context propagation.
"""

from typing import List, Optional

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Logging import DEFAULT_LOGGER as log

from .Matchers import Match, NonBranchingMatcher, SubgraphMatcher


class _MemoReach():
    """Internal helper for computing reachable nodes in a graph.

    This class uses memoization to efficiently compute the set of nodes
    that are reachable from a set of input tensors and that reach a set
    of output tensors. Used for pattern replacement to determine which
    nodes can be safely removed.
    """

    def __init__(self, graph, inputTensors, outputTensors):
        self.memo = {}
        self.graph = graph
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors

    def reachingSet(self):
        reachingSet = []
        for inTensor in self.inputTensors:
            for user in inTensor.outputs:
                reachingSet += self._reachingSet(user)

        nodeNames = {node.name for node in reachingSet}
        retList = [node for node in self.graph.nodes if node.name in nodeNames]
        return retList

    def _reachingSet(self, node: gs.Node) -> List[gs.Node]:

        if node.name in self.memo.keys():
            return self.memo[node.name]

        reachingSet = []

        if any([output in self.outputTensors for output in node.outputs]):
            self.memo[node.name] = [node]
            return [node]

        oSet = []
        for outp in node.outputs:
            for out in outp.outputs:
                if outp not in self.inputTensors:
                    oSet.append(out)

        for potentialNode in oSet:
            if potentialNode.name in self.memo.keys():
                nodeRet = self.memo[potentialNode.name]
            else:
                nodeRet = self._reachingSet(potentialNode)

            reachingSet += nodeRet

        if reachingSet != []:
            reachingSet.append(node)

        self.memo[node.name] = reachingSet

        return reachingSet


@gs.Graph.register()
def deleteNode(self, node: gs.Node):
    # LMACAN: Assume only one input and only one output tensor

    inputTensor = node.inputs[0]
    outputTensor = node.outputs[0]

    isGlobalOutputTensor = len(outputTensor.outputs) == 0

    if isGlobalOutputTensor:
        inputTensor.name = outputTensor.name  # Preserve the output tensor name
        outputTensor.name = outputTensor.name + "_throwaway"  # Avoid same named tensors in graph; gets immediately removed with cleanup
        self.outputs[self.outputs.index(outputTensor)] = inputTensor
    else:
        for outputNode in list(outputTensor.outputs):
            # Swap the outputTensor with inputTensor in the downstream nodes
            outputNode.inputs[outputNode.inputs.index(outputTensor)] = inputTensor
        node.inputs.clear()
        node.outputs.clear()

    self.cleanup()


def _reachableNodes(graph: gs.Graph, inputTensors: List[gs.Tensor], outputTensors: List[gs.Tensor]) -> List[gs.Node]:

    _inputTensors = [tensor for tensor in inputTensors.copy() if tensor.name in graph.tensors().keys()]
    _outputTensors = [tensor for tensor in outputTensors.copy() if tensor.name in graph.tensors().keys()]

    retList = _MemoReach(graph, _inputTensors, _outputTensors).reachingSet()

    return retList


@gs.Graph.register()
def replaceInsertNode(self, inputs, outputs, newNode):
    reachableSet = _reachableNodes(self, inputs, outputs)

    ret = self.layer(op = newNode.op, name = newNode.name, attrs = newNode.attrs, inputs = inputs, outputs = outputs)

    for node in reachableSet:
        node.outputs = []

    self.toposort().cleanup()


class Pass():
    """Base class for all optimization passes.

    Pass provides the foundation for graph transformation passes with support for:
    - Hierarchical composition via subpasses
    - Automatic subpass registration
    - Parent-child relationships for pass hierarchies

    Subclasses should implement run_pass() and be decorated with @contextaware
    or @contextagnostic to add the appropriate execution MixIn.

    Attributes:
        parent: Reference to parent pass if this is a subpass
        _subpasses: Dictionary of registered subpasses by name

    Usage:
        Subclasses should NOT override __init__ unless calling super().__init__().
        Instead, implement run_pass() and use decorators:

        @contextagnostic
        class MyPass(Pass):
            def run_pass(self, graph: gs.Graph) -> gs.Graph:
                # Transform graph
                return graph
    """

    def __init__(self):
        self.parent = None
        self._subpasses = {}

    def __setattr__(self, attribute, value):
        """Automatically register Pass instances as subpasses when assigned as attributes."""
        if isinstance(value, Pass) and attribute != 'parent':
            self.register_subpass(attribute, value)
        super(Pass, self).__setattr__(attribute, value)

    def register_subpass(self, name, value):
        """Register a subpass with a given name.

        Args:
            name: Name for the subpass (used for lookup)
            value: The Pass instance to register
        """
        if name in self._subpasses.keys():
            del self._subpasses[name]

        value.parent = self
        self._subpasses[name] = value

    def remove_subpass(self, name):
        """Remove a registered subpass by name.

        Args:
            name: Name of the subpass to remove

        Raises:
            KeyError: If no subpass with the given name exists
            AttributeError: If called before Pass.__init__()
        """
        try:
            del self._subpasses[name]
        except KeyError:
            log.error(f"No subpass with name {name}, cannot remove!")
        except AttributeError:
            raise AttributeError("Cannot remove sub-pass before calling Pass.__init__!")

    def __getattr__(self, attribute):
        """Allow attribute-style access to registered subpasses."""
        if self._subpasses is not None and attribute in self._subpasses.keys():
            return self._subpasses[attribute]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attribute}")

    def named_subpasses(self):
        """Get a copy of the subpasses dictionary.

        Returns:
            Dictionary mapping subpass names to Pass instances
        """
        return self._subpasses.copy()


class ContextAwarePassMixIn():
    """MixIn for passes that need access to NetworkContext.

    Context-aware passes can access buffer information, type information,
    memory allocation state, and other network-level metadata during
    transformation. This is essential for passes that need to:
    - Create new buffers
    - Query type information
    - Check memory constraints
    - Access global network state

    This MixIn is automatically added via the @contextaware decorator.
    """

    # DO NOT OVERWRITE this function in custom pass subclasses unless you have
    # a very good reason!
    def apply(self, ctxt, graph):
        """Apply the pass to a graph with context.

        This orchestrates retargeting (for graph-specific passes) and
        then runs the actual transformation.
        """
        ctxt, graph = self.retarget(ctxt, graph)
        ctxt, graph = self.run_pass(ctxt, graph)
        return ctxt, graph

    def __call__(self, ctxt: NetworkContext, graph: gs.Graph):
        return self.apply(ctxt, graph)

    # overwrite this if your pass is specific to a graph instance (e.g., most
    # "dynamic" SequentialPass derivatives will be, as the list of passes to
    # execute probably depends on the graph. See e.g.
    # ReplaceSequentialPatternPass for an example)
    def retarget(self, ctxt: NetworkContext, graph: gs.Graph):
        """Retarget the pass to a new graph instance.

        Override this for passes that cache graph-specific information
        (e.g., pattern matches). The default implementation is a no-op.

        Args:
            ctxt: Network context
            graph: New graph to target

        Returns:
            Tuple of (potentially modified context, graph)
        """
        return ctxt, graph


class ContextAgnosticPassMixIn():
    """MixIn for passes that only transform the graph structure.

    Context-agnostic passes work purely on the ONNX graph structure without
    needing access to buffer information, types, or other network metadata.
    These passes are typically used for:
    - Graph rewrites (e.g., operator fusion)
    - Constant folding
    - Dead code elimination
    - Pattern-based transformations

    This MixIn is automatically added via the @contextagnostic decorator.
    """

    # DO NOT OVERWRITE this function in custom pass subclasses unless you have
    # a very good reason!
    def apply(self, graph: gs.Graph) -> gs.Graph:
        """Apply the pass to a graph.

        This orchestrates retargeting (for graph-specific passes) and
        then runs the actual transformation.
        """
        graph = self.retarget(graph)
        graph = self.run_pass(graph)
        return graph

    def __call__(self, graph: gs.Graph):
        return self.apply(graph)

    # overwrite this if your pass is specific to a graph instance (e.g., most
    # "dynamic" SequentialPass derivatives will be, as the list of passes to
    # execute probably depends on the graph. See e.g.
    # ReplaceSequentialPatternPass for an example)
    def retarget(self, graph: gs.Graph) -> gs.Graph:
        """Retarget the pass to a new graph instance.

        Override this for passes that cache graph-specific information
        (e.g., pattern matches). The default implementation is a no-op.

        Args:
            graph: New graph to target

        Returns:
            The graph (potentially retargeted)
        """
        return graph


class ContextAwareSequentialPassMixIn(ContextAwarePassMixIn):

    def run_pass(self, ctxt: NetworkContext, graph: gs.Graph):
        for p in self.named_subpasses().values():
            ctxt, graph = p.apply(ctxt, graph)
        return ctxt, graph


class ContextAgnosticSequentialPassMixIn(ContextAgnosticPassMixIn):

    def run_pass(self, graph: gs.Graph):
        for p in self.named_subpasses().values():
            graph = p.apply(graph)
        return graph


class SequentialPass(Pass):
    """A pass that executes a sequence of subpasses in order.

    SequentialPass is the primary composition mechanism for building complex
    optimization pipelines from simpler passes. Subpasses are executed in the
    order they were provided, with the output of each pass feeding into the next.

    Args:
        *passes: Variable number of Pass instances to execute in sequence
        name_prefix: Optional prefix for subpass names (default: '')

    Example:
        # Create a pipeline of optimizations
        optimizer = SequentialPass(
            ConstantFoldingPass(),
            DeadCodeEliminationPass(),
            OperatorFusionPass(),
            name_prefix='lowering'
        )

        # The passes will be registered as:
        # lowering_0: ConstantFoldingPass
        # lowering_1: DeadCodeEliminationPass
        # lowering_2: OperatorFusionPass

    Note:
        SequentialPass itself is abstract - use @contextaware or @contextagnostic
        decorator to create a usable version. The decorator adds the appropriate
        MixIn that orchestrates sequential execution.
    """

    def __init__(self, *passes, name_prefix = ''):
        super(SequentialPass, self).__init__()
        self.name_prefix = name_prefix
        self.setup_passes(passes)

    def setup_passes(self, passes):
        """Register all provided passes as numbered subpasses.

        Args:
            passes: Tuple/list of Pass instances to register
        """
        for i, p in enumerate(passes):
            self.register_subpass(self.name_prefix + '_' + str(i), p)


class ContextAwareReplaceMatchWithModulePassMixIn(ContextAwarePassMixIn):

    def run_pass(self, ctxt: NetworkContext, graph: gs.Graph):
        if self.replacementNode is not None:
            graph.replaceInsertNode(self.replacementNode)
        return ctxt, graph


class ContextAgnosticReplaceMatchWithModulePassMixIn(ContextAgnosticPassMixIn):

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        if self.replacementNode is not None:
            graph.replaceInsertNode(self.replacementNode)
        return graph


class ReplaceMatchWithModulePass(Pass):
    #Matches are specific to graph instances, so don't use this type of pass on its
    #own if you want to reuse it!
    def __init__(self, match: Match, module: gs.Node):
        # this class needs a name field because the inserted submodules will be named
        super(ReplaceMatchWithModulePass, self).__init__()
        self.match = match
        self.replacementNode = module


class ContextAwareReplaceSequentialPatternPassMixIn(ContextAwareSequentialPassMixIn):

    def retarget(self, ctxt: NetworkContext, graph: gs.Graph):
        # to retarget to a new graph, clear all registered subpasses.
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)
        self.matches = self.matcher.match(graph, self.pattern)
        for i, m in enumerate(self.matches):
            ctxt, graph = self.replacement_fn(ctxt, graph, m, f"{self.name}_{i}", **self.kwargs)
        graph.cleanup().toposort()
        return ctxt, graph


class ContextAgnosticReplaceSequentialPatternPassMixIn(ContextAgnosticSequentialPassMixIn):

    def retarget(self, graph: gs.Graph):
        # to retarget to a new graph, clear all registered subpasses.
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)
        self.matches = self.matcher.match(graph, self.pattern)
        for i, m in enumerate(self.matches):
            graph = self.replacement_fn(graph, m, f"{self.name}_{i}", **self.kwargs)
        graph.cleanup().toposort()
        return graph


class ReplaceSequentialPatternPass(SequentialPass):
    """A pass that finds and replaces all instances of a graph pattern.

    This is one of the most powerful transformation mechanisms in Deeploy.
    It searches the graph for all occurrences of a specific subgraph pattern
    and replaces each match with new nodes generated by a replacement function.

    The pass is "sequential" because it creates a subpass for each match found,
    allowing each replacement to be tracked independently.

    Args:
        pattern: An ONNX graph representing the pattern to search for
        replacement_fn: Callable that generates replacement nodes for each match.
                       Signature depends on context-awareness:
                       - Context-aware: fn(ctxt, graph, match, name, **kwargs)
                       - Context-agnostic: fn(graph, match, name, **kwargs)
        name: Name for this pass (used as prefix for subpass names)
        matcher: SubgraphMatcher instance for pattern matching (default: NonBranchingMatcher)
        **kwargs: Additional arguments passed to replacement_fn

    Example (Context-Agnostic):
        # Define a pattern: Transpose -> MatMul -> Transpose
        pattern = create_pattern_graph(["Transpose", "MatMul", "Transpose"])

        # Define replacement function
        def replace_with_optimized(graph, match, name):
            # Create optimized MatMul without transposes
            return gs.Node(op="MatMul", name=name, ...)

        # Create the pass
        pass = ReplaceSequentialPatternPass(
            pattern=pattern,
            replacement_fn=replace_with_optimized,
            name="OptimizeMatMul"
        )

        # Apply to graph (finds all matches and replaces them)
        graph = pass(graph)

    Example (Context-Aware):
        @contextaware
        class FuseConvBatchNormPass(ReplaceSequentialPatternPass):
            def __init__(self):
                super().__init__(
                    pattern=create_conv_bn_pattern(),
                    replacement_fn=self.fuse_conv_bn,
                    name="FuseConvBN"
                )

            def fuse_conv_bn(self, ctxt, graph, match, name):
                # Access context for buffer information
                conv_weights = ctxt.lookup(match.nodes[0].inputs[1].name)
                # Create fused operator
                return gs.Node(...)

    How it Works:
        1. **Retargeting**: When applied to a graph, retarget() finds all pattern matches
        2. **Replacement**: For each match, calls replacement_fn to generate new nodes
        3. **Subpass Creation**: Each replacement is registered as a numbered subpass
        4. **Execution**: Subpasses execute sequentially via SequentialPass mechanism

    Note:
        This pass requires retargeting for each new graph because pattern matches
        are graph-specific. The retarget() method clears old subpasses and regenerates
        them based on the new graph's matches.
    """

    def __init__(self,
                 pattern: gs.Graph,
                 replacement_fn: callable,
                 name: str,
                 matcher: Optional[SubgraphMatcher] = None,
                 **kwargs):
        super().__init__(name_prefix = name)
        self.pattern = pattern
        self.matcher = matcher
        if matcher is None:
            self.matcher = NonBranchingMatcher()
        self.replacement_fn = replacement_fn
        self.name = name
        self.kwargs = kwargs


def contextagnostic(cls):
    """Decorator to make a Pass subclass context-agnostic.

    This decorator dynamically adds the appropriate ContextAgnosticMixIn to a Pass
    subclass, enabling it to transform graphs without needing NetworkContext.
    The decorator automatically selects the right MixIn based on the pass type.

    Usage:
        @contextagnostic
        class MyOptimizationPass(Pass):
            def run_pass(self, graph: gs.Graph) -> gs.Graph:
                # Transform graph structure only
                return graph

        # Can now call as: graph = MyOptimizationPass()(graph)

    Args:
        cls: The Pass subclass to decorate

    Returns:
        A new class with the appropriate ContextAgnosticMixIn added

    Raises:
        Exception: If cls is not a subclass of Pass

    Note:
        The decorator checks pass type in order from most specific to least specific:
        1. ReplaceMatchWithModulePass → ContextAgnosticReplaceMatchWithModulePassMixIn
        2. ReplaceSequentialPatternPass → ContextAgnosticReplaceSequentialPatternPassMixIn
        3. SequentialPass → ContextAgnosticSequentialPassMixIn
        4. Pass → ContextAgnosticPassMixIn
    """
    mixinClass = None
    # These need to be sorted from most specific parent class to least specific parent class!
    if issubclass(cls, ReplaceMatchWithModulePass):
        mixinClass = ContextAgnosticReplaceMatchWithModulePassMixIn
    elif issubclass(cls, ReplaceSequentialPatternPass):
        mixinClass = ContextAgnosticReplaceSequentialPatternPassMixIn
    elif issubclass(cls, SequentialPass):
        mixinClass = ContextAgnosticSequentialPassMixIn
    elif issubclass(cls, Pass):
        mixinClass = ContextAgnosticPassMixIn
    else:
        raise Exception(f"Tried to decorate class {cls} as contextagnostic, but failed!")
    return type(cls.__name__, (cls, mixinClass), {})


def contextaware(cls):
    """Decorator to make a Pass subclass context-aware.

    This decorator dynamically adds the appropriate ContextAwareMixIn to a Pass
    subclass, enabling it to access NetworkContext during transformation. The
    decorator automatically selects the right MixIn based on the pass type.

    Usage:
        @contextaware
        class MyBufferCreatingPass(Pass):
            def run_pass(self, ctxt: NetworkContext, graph: gs.Graph):
                # Can access context to create buffers, query types, etc.
                buffer = ctxt.hoistConstant(...)
                return ctxt, graph

        # Can now call as: ctxt, graph = MyBufferCreatingPass()(ctxt, graph)

    Args:
        cls: The Pass subclass to decorate

    Returns:
        A new class with the appropriate ContextAwareMixIn added

    Raises:
        Exception: If cls is not a subclass of Pass

    Note:
        The decorator checks pass type in order from most specific to least specific:
        1. ReplaceMatchWithModulePass → ContextAwareReplaceMatchWithModulePassMixIn
        2. ReplaceSequentialPatternPass → ContextAwareReplaceSequentialPatternPassMixIn
        3. SequentialPass → ContextAwareSequentialPassMixIn
        4. Pass → ContextAwarePassMixIn
    """
    mixinClass = None
    # These need to be sorted from most specific parent class to least specific parent class!
    if issubclass(cls, ReplaceMatchWithModulePass):
        mixinClass = ContextAwareReplaceMatchWithModulePassMixIn
    elif issubclass(cls, ReplaceSequentialPatternPass):
        mixinClass = ContextAwareReplaceSequentialPatternPassMixIn
    elif issubclass(cls, SequentialPass):
        mixinClass = ContextAwareSequentialPassMixIn
    elif issubclass(cls, Pass):
        mixinClass = ContextAwarePassMixIn
    else:
        raise Exception(f"Tried to decorate class {cls} as contextaware, but failed!")
    return type(cls.__name__, (cls, mixinClass), {})
