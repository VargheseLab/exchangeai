__all__ = [
    'OnnxGlobalMaxPool',
    'OnnxGlobalMaxPoolWithKnownInputShape',
]

from typing import List

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult, get_shape_from_value_info, onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx, OnnxToTorchModuleWithCustomExport

import torch
import torch.nn as nn

class OnnxGlobalMaxPool(nn.Module, OnnxToTorchModuleWithCustomExport):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        def _forward():
            x_dims = torch.tensor(range(2, len(input_tensor.shape)))
            result_tensor = input_tensor
            for dim in x_dims:
                result_tensor, _ = torch.max(result_tensor, dim, keepdim=True)
            return result_tensor

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(_forward, 'GlobalMaxPool', input_tensor, {})

        return _forward()


class OnnxGlobalMaxPoolWithKnownInputShape(nn.Module, OnnxToTorchModuleWithCustomExport):
    def __init__(self, input_shape: List[int]):
        super().__init__()
        self._x_dims = torch.tensor(range(2, len(input_shape)))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            result_tensor = input_tensor
            for dim in self._x_dims:
                result_tensor, _ = torch.max(result_tensor, dim, keepdim=True)
            return result_tensor

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(_forward, 'GlobalMaxPool', input_tensor, {})

        return _forward()


@add_converter(operation_type='GlobalMaxPool', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    input_value_info = graph.value_info[node.input_values[0]]
    input_shape = get_shape_from_value_info(input_value_info)

    if input_shape is not None:
        torch_module = OnnxGlobalMaxPoolWithKnownInputShape(input_shape=input_shape)
    else:
        torch_module = OnnxGlobalMaxPool()

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )