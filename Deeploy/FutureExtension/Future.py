# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Future Extension - Asynchronous Execution Support

This module implements the Future/Promise pattern for Deeploy, enabling asynchronous
execution of operations with deferred result resolution. This is critical for:

- Overlapping computation with data movement (DMA transfers)
- Pipeline parallelism across heterogeneous cores
- Non-blocking accelerator dispatch

Architecture:
    The Future abstraction wraps a pointer to a result that will be computed
    asynchronously. It maintains a reference to a state structure that tracks
    the operation's completion status. Before accessing the future's value,
    the resolve check ensures the operation has completed.

Key Concepts:
    - Future: A pointer to data that will be available in the future
    - State Reference: A structure tracking the completion state of the async operation
    - Resolve Check: A template that generates code to wait for operation completion
    - Dispatch Check: A template that generates code to check if dispatch is safe

Example Usage:
    A DMA transfer might return a Future[Buffer], allowing the caller to continue
    execution while the transfer completes. Before reading the buffer, a resolve
    check ensures the DMA has finished.
"""

from typing import Optional, Type

from Deeploy.AbstractDataTypes import BaseType, Pointer
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, StructBuffer


class Future(Pointer):
    """A Future represents a pointer to data that will be available asynchronously.

    Futures extend Pointer with state tracking for asynchronous operations. They
    maintain a reference to a state structure that indicates operation completion,
    and provide templates for generating synchronization code.

    Attributes:
        stateReference: Reference to the state structure tracking operation status
        stateReferenceType: Type of the state structure (class attribute)
        resolveCheckTemplate: Template for generating completion wait code (class attribute)
        dispatchCheckTemplate: Template for generating pre-dispatch safety checks (class attribute)

    The state reference is typically a hardware-specific structure (e.g., DMA status
    register, accelerator completion flag) that indicates when the operation has finished.
    """

    __slots__ = ['stateReference']
    stateReferenceType: Type[Pointer]
    resolveCheckTemplate: NodeTemplate
    dispatchCheckTemplate: NodeTemplate

    def assignStateReference(self, stateReference: StructBuffer, ctxt: Optional[NetworkContext] = None):
        """Assign a state structure to track this Future's completion status.

        The state reference is a platform-specific structure (e.g., DMA control block,
        accelerator status register) that indicates when the asynchronous operation
        has completed. This method validates that the provided structure matches
        the expected type before assignment.

        Args:
            stateReference: The state tracking structure to assign
            ctxt: Optional network context for type checking

        Raises:
            Exception: If the stateReference type is incompatible with stateReferenceType

        Example:
            # For a DMA future, assign DMA channel status structure
            dma_future.assignStateReference(dma_channel_status, network_context)
        """
        if self.stateReferenceType.checkPromotion(stateReference.structDict, ctxt):  # type: ignore
            self.stateReference = stateReference
        else:
            raise Exception(f"Can't assign {stateReference} to {self}!")

    def _bufferRepresentation(self):
        """Generate the buffer representation for code generation.

        Returns:
            Dict containing the state reference name for template substitution.
            This is used during code generation to reference the state structure
            in the generated C code.
        """
        return {"stateReference": self.stateReference.name}


def FutureClass(underlyingType: BaseType, stateReferenceType: Type[Pointer], resolveCheckTemplate: NodeTemplate,
                dispatchCheckTemplate: NodeTemplate) -> Type[Future]:
    """Factory function to create specialized Future subclasses for different async operations.

    This factory dynamically creates Future subclasses tailored to specific asynchronous
    operations (e.g., DMA transfers, accelerator dispatches). Each Future type is associated
    with:
    - The underlying data type being computed asynchronously
    - A state reference type for tracking completion
    - Templates for generating synchronization code

    The factory uses memoization - if a Future class for the given state reference type
    already exists, it returns the cached version instead of creating a duplicate.

    Args:
        underlyingType: The base type of data the Future will eventually point to
                       (e.g., int8_t for a buffer of 8-bit integers)
        stateReferenceType: Type of the structure tracking operation completion status
                           (e.g., DmaChannelState, AcceleratorStatus)
        resolveCheckTemplate: NodeTemplate that generates code to wait for operation completion.
                             This ensures the Future's value is ready before access.
        dispatchCheckTemplate: NodeTemplate that generates code to check if it's safe to
                              dispatch a new operation (e.g., DMA channel available)

    Returns:
        A Future subclass specialized for the specified async operation type.
        The class is cached in globals for reuse.

    Example:
        # Create a Future type for DMA transfers of int8 buffers
        DmaFuture = FutureClass(
            underlyingType=int8_t,
            stateReferenceType=DmaChannelState,
            resolveCheckTemplate=DmaWaitTemplate,
            dispatchCheckTemplate=DmaReadyTemplate
        )

        # Now can instantiate futures for DMA operations
        buffer_future = DmaFuture(...)

    Note:
        The generated Future class name follows the pattern: "{stateReferenceType.typeName}Future"
        e.g., "DmaChannelStateFuture"
    """
    typeName = stateReferenceType.typeName + "Future"
    if typeName not in globals().keys():
        # Dynamically create a new Future subclass with the specified properties
        retCls = type(
            typeName, (Future,), {
                "typeName": underlyingType.typeName + "*",  # Future is a pointer to the underlying type
                "typeWidth": 32,  # Pointers are 32-bit on target platforms
                "referencedType": underlyingType,
                "stateReferenceType": stateReferenceType,
                "resolveCheckTemplate": resolveCheckTemplate,
                "dispatchCheckTemplate": dispatchCheckTemplate
            })
        globals()[typeName] = retCls  # Cache for reuse
    else:
        # Return cached class if already created
        retCls = globals()[typeName]

    return retCls
