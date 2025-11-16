# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Engine Coloring Extension - Hardware Accelerator Scheduling

This module implements "engine coloring" - a compiler pass that assigns operations
to specific hardware execution engines (accelerators, cores, or functional units)
on heterogeneous SoCs.

Engine Coloring Concept:
    Heterogeneous platforms contain multiple execution engines (e.g., CPU cores,
    DSP accelerators, neural network accelerators). Engine coloring determines
    which engine should execute each operation by:

    1. Analyzing operation characteristics (compute type, data types)
    2. Matching operations to capable engines based on operation mappings
    3. Considering resource constraints and optimization objectives
    4. Annotating each ONNX node with its assigned "engine" attribute

Architecture:
    - EngineMapper: Implements the mapping algorithm from operations to engines
    - EngineColoringPass: Optimization pass that performs the coloring
    - EngineColoringDeployer: Deployer that integrates coloring into compilation

Example Scenario:
    On a platform with a CPU and a matrix multiplication accelerator:
    - MatMul/GEMM operations → Accelerator engine
    - Activation functions → CPU engine
    - Control flow → CPU engine

This enables optimal utilization of heterogeneous hardware capabilities.
"""

from typing import Callable, Dict, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.NetworkDeployerWrapper import NetworkDeployerWrapper
from Deeploy.DeeployTypes import DeploymentEngine, DeploymentPlatform, NetworkDeployer, Schedule, TopologyOptimizer
from Deeploy.EngineExtension.OptimizationPasses.TopologyOptimizationPasses.EngineColoringPasses import \
    EngineColoringPass, EngineMapper


class EngineColoringDeployer(NetworkDeployer):
    """Network deployer that assigns operations to hardware execution engines.

    This deployer extends the base NetworkDeployer with engine coloring capabilities,
    ensuring that every operation in the computation graph is assigned to a compatible
    hardware execution engine before code generation.

    The deployer interleaves engine coloring passes with other lowering passes to
    handle newly created nodes from graph transformations.

    Attributes:
        engineDict: Mapping from engine names to DeploymentEngine objects
        engineMapperCls: The EngineMapper class used for the coloring algorithm
    """

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable[[gs.Graph], Schedule] = lambda graph: list(graph.nodes),
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 engineMapperCls: Type[EngineMapper] = EngineMapper):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir)
        self._initEngineColoringDeployer(engineMapperCls)

    def _initEngineColoringDeployer(self, engineMapperCls: Type[EngineMapper]):
        """Initialize the engine coloring infrastructure.

        This method sets up the engine coloring system by:
        1. Creating a dictionary of available engines from the platform
        2. Instantiating the engine mapper with the available engines
        3. Interleaving engine coloring passes with existing lowering passes

        The interleaving strategy ensures that operations created during graph
        transformations are also assigned to engines.

        Args:
            engineMapperCls: The EngineMapper class to use for mapping operations to engines
        """
        # Build engine dictionary from platform definition
        self.engineDict = {engine.name: engine for engine in self.Platform.engines}

        # Create mapper instance to perform the operation→engine assignment
        engineMapper = engineMapperCls(self.engineDict)
        engineColoringPass = EngineColoringPass(engineMapper)

        # Interleave coloring with existing passes: [color, pass1, color, pass2, color, ...]
        # This ensures newly created nodes from transformations get colored
        loweringPasses = [engineColoringPass]
        for _pass in self.loweringOptimizer.passes:
            loweringPasses.append(_pass)
            loweringPasses.append(engineColoringPass)
        self.loweringOptimizer.passes = loweringPasses

    def lower(self, graph: gs.Graph) -> gs.Graph:
        """Lower the graph with engine coloring validation.

        Performs standard graph lowering and validates that all nodes have been
        assigned to an execution engine. This is a critical check to ensure the
        graph is ready for code generation.

        Args:
            graph: The ONNX graph to lower

        Returns:
            The lowered graph with all nodes colored

        Raises:
            AssertionError: If any nodes lack engine assignments after lowering
        """
        graph = super().lower(graph)

        # Validate that all nodes have been assigned to an engine
        uncoloredNodes = [node for node in graph.nodes if "engine" not in node.attrs]
        uncoloredOperations = set(node.op for node in uncoloredNodes)
        assert len(
            uncoloredNodes
        ) == 0, f"Missing engine color for nodes {[node.name for node in uncoloredNodes]} with operations {uncoloredOperations}"
        return graph

    def _selectEngine(self, node: gs.Node) -> DeploymentEngine:
        """Select the execution engine assigned to a node.

        Retrieves the DeploymentEngine object for a node based on its "engine"
        attribute, validating that:
        1. The node has an engine assignment
        2. The engine name is valid
        3. The engine supports the node's operation

        Args:
            node: The ONNX node to get the engine for

        Returns:
            The DeploymentEngine object assigned to this node

        Raises:
            AssertionError: If the node lacks an engine, has an invalid engine,
                          or the engine doesn't support the operation
        """
        assert "engine" in node.attrs, f"Node {node.name} doesn't have an engine color."
        engineName = node.attrs["engine"]
        assert isinstance(engineName, str) and engineName in self.engineDict, \
            f"Node {node.name} has an invalid engine {engineName} assigned."
        engine = self.engineDict[engineName]
        assert node.op in engine.Mapping, f"No mapping found for {node.op} in engine {engine.name}"
        return engine


class EngineColoringDeployerWrapper(EngineColoringDeployer, NetworkDeployerWrapper):
    """Wrapper that adds engine coloring capabilities to an existing deployer.

    This class allows you to augment any NetworkDeployer with engine coloring
    functionality without modifying the original deployer class. It uses the
    decorator pattern to wrap an existing deployer and add the engine assignment
    infrastructure.

    Example:
        # Add engine coloring to an existing deployer
        base_deployer = SignPropDeployer(...)
        colored_deployer = EngineColoringDeployerWrapper(base_deployer)

        # Now the deployer will assign operations to engines
        colored_deployer.prepare()
    """

    def __init__(self, deployer: NetworkDeployer, engineMapperCls: Type[EngineMapper] = EngineMapper) -> None:
        """Initialize the wrapper with an existing deployer.

        Args:
            deployer: The NetworkDeployer to wrap with engine coloring
            engineMapperCls: The EngineMapper class to use for mapping (default: EngineMapper)
        """
        NetworkDeployerWrapper.__init__(self, deployer)
        self._initEngineColoringDeployer(engineMapperCls)
