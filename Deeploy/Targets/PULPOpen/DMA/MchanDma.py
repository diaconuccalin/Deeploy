# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import CodeSnippet, NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DirectionWaitingStrategy, DmaDirection, Future


class MchanChannelFuture(Future):

    _initTemplate = NodeTemplate("uint32_t ${name} = (uint32_t) -1;")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("${name} = mchan_channel_alloc();")

    _waitTemplate = NodeTemplate("""
if (${name} <= MCHAN_CHANNEL_ID_MAX) {
    mchan_channel_wait(${name});
    mchan_channel_free(${name});
}
""")


class MchanDma(AsyncDma):

    _transferTemplates = {
        1: NodeTemplate("mchan_transfer_1d(${cmd}, ${loc}, ${ext});"),
        2: NodeTemplate("mchan_transfer_2d_ext_strided(${cmd}, ${loc}, ${ext}, ${size_1d}, ${stride_2d});"),
    }
    # For tensors whose external buffer exceeds the 17-bit length field: split at runtime (see
    # mchan_v7.h). Chosen per tensor so all tiles of a tensor share the same call structure.
    _chunkedTransferTemplates = {
        1:
            NodeTemplate("mchan_transfer_1d_chunked(${cmd_flags}, ${size}, ${loc}, ${ext});"),
        2:
            NodeTemplate("mchan_transfer_2d_ext_strided_chunked(${cmd_flags}, ${size}, ${loc}, ${ext}, "
                         "${size_1d}, ${stride_2d});"),
    }
    _waitingStrategy = DirectionWaitingStrategy(MchanChannelFuture, "channel")

    MAX_TRANSFER_SIZE = (1 << 17) - 1  # bytes representable in the mchan v7 length field

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def transfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                 shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                 direction: DmaDirection, future: Future) -> List[CodeSnippet]:
        if math.prod(externalBuffer.shape) <= self.MAX_TRANSFER_SIZE:
            return super().transfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future)

        self.checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        if len(shape) == 2:
            assert shape[1] <= self.MAX_TRANSFER_SIZE, f"Mchan cannot split 2D transfers with {shape[1]}-byte rows"
        opRepr = self.transferOpRepr(externalBuffer,
                                     localBuffer,
                                     shape,
                                     strideExt,
                                     strideLoc,
                                     direction,
                                     future,
                                     checkSize = False)
        opRepr["size"] = math.prod(shape)
        opRepr["cmd_flags"] = opRepr["cmd"] - opRepr["size"]
        del opRepr["cmd"]
        return [CodeSnippet(self._chunkedTransferTemplates[len(shape)], opRepr)]

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)

        transferRank = len(shape)
        assert strideExt[
            -1] == 1, "Mchan supports only contigous transfers of the innermost dimension for external memory"
        if transferRank == 1:
            assert strideLoc[0] == 1, "Mchan supports only contigous transfers for local memory"
        else:
            assert strideLoc[0] == shape[1] and strideLoc[
                1] == 1, "Mchan supports only contigous transfers for local memory"

    def transferOpRepr(self,
                       externalBuffer: VariableBuffer,
                       localBuffer: VariableBuffer,
                       shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...],
                       strideLoc: Tuple[int, ...],
                       direction: DmaDirection,
                       future: Future,
                       checkSize: bool = True) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)

        transferRank = len(shape)

        mchanFlags = 0
        mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0  # direction
        mchanFlags += (1 << 1)  # increment addresses
        mchanFlags += (1 << 2) if transferRank == 2 else 0  # 2d transfer
        mchanFlags += (1 << 3)  # event enable

        mchanTransferSize = math.prod(shape)
        # The length field is 17 bits wide: sizes >= 2**17 would overflow into the flag bits
        # (e.g. 2**17 sets the direction flag). Tensors with larger buffers use the chunked transfers.
        assert not checkSize or 0 < mchanTransferSize <= self.MAX_TRANSFER_SIZE, (
            "The transfer size is not representable with 17 bits. "
            f"Received transfer size {mchanTransferSize} > {self.MAX_TRANSFER_SIZE}")

        operatorRepresentation["cmd"] = (mchanFlags << 17) + mchanTransferSize

        if transferRank == 2:
            operatorRepresentation["size_1d"] = shape[1]
            operatorRepresentation["stride_2d"] = strideExt[0]

        return operatorRepresentation
