import os
import os.path
from typing import Optional, Tuple, Union

from networkx import Graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor

from .attention import EfficientAttention

from .. import graph as g
from ..graph import Node

KernelSize = Union[int, Tuple[int, int]]
StrideSize = Union[int, Tuple[int, int]]


class ConvELUBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: KernelSize, stride: StrideSize) -> None:

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

        padding_size = (kernel_size - 1) // 2
        self.padding = tuple([padding_size] * 4)

    def forward(self, x: Tensor) -> Tensor:
        x_padded = F.pad(x, self.padding)
        return self.layers(x_padded)


class NodeBlock(nn.Module):

    def __init__(self, node: Node, in_channels: int, out_channels: int,
                 kernel_size: KernelSize) -> None:

        super().__init__()

        self.numberof_inputs = len(node.inputs)

        initial_means = torch.ones(self.numberof_inputs)
        self.mean_weight = nn.Parameter(initial_means) \
            if self.numberof_inputs > 1 else None

        if node.node_type == 'input':
            stride = 2
        else:
            in_channels = out_channels
            stride = 1

        self.convolution = ConvELUBlock(in_channels, out_channels,
                                        kernel_size, stride=stride)

    def resize(self, x: Tensor, desired_size: Size) -> Tensor:
        _, _, input_h, input_w = x.size()
        _, _, desired_h, desired_w = desired_size

        dw = desired_w - input_w
        dh = desired_h - input_h

        pad_size = [
            dw // 2, dw - dw // 2,
            dh // 2, dh - dh // 2
        ]

        return F.pad(x, pad_size, mode='reflect')

    def forward(self, *inputs) -> Tensor:
        if self.numberof_inputs > 1:
            out = torch.sigmoid(self.mean_weight[0]) * inputs[0]

            for i, x in enumerate(inputs[1:]):
                if x.size(2) != out.size(2):
                    x = self.resize(x, out.size())

                out += torch.sigmoid(self.mean_weight[i]) * x
        else:
            out = inputs[0]

        return self.convolution(out)


class GraphBlock(nn.Module):

    def __init__(self, graph: Graph, in_channels: int, out_channels: int,
                 kernel_size: KernelSize) -> None:

        super().__init__()

        self.nodes, self.in_nodes, self.out_nodes = g.get_graph_info(graph)

        self.node_blocks = nn.ModuleList()

        for node in self.nodes:
            block = NodeBlock(node, in_channels, out_channels, kernel_size)
            self.node_blocks.append(block)

    def resize(self, x: Tensor, desired_size: Size) -> Tensor:
        _, _, output_h, output_w = x.size()
        _, _, desired_h, desired_w = desired_size

        dw = desired_w - output_w
        dh = desired_h - output_h

        pad_size = [
            dw // 2, dw - (dw // 2),
            dh // 2, dh - (dh // 2)
        ]

        return F.pad(x, pad_size, mode='reflect')

    def forward(self, x: Tensor) -> Tensor:
        results = {idx: self.node_blocks[idx](x) for idx in self.in_nodes}

        for idx, node in enumerate(self.nodes):
            if idx in self.in_nodes:
                continue

            inputs = [results[input_id] for input_id in node.inputs]
            results[idx] = self.node_blocks[idx](*inputs)

        for i, idx in enumerate(self.out_nodes):
            if i == 0:
                out = results[idx]
                continue

            if out.size(2) != results[idx].size(2):
                results[idx] = self.resize(results[idx], out.size())

            out += results[idx]

        return out / len(self.out_nodes)


class EncoderStage(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: KernelSize, stage: int, heads: int = 8,
                 nodes: int = 5, p: float = 0.75, k: int = 4,
                 seed: Optional[int] = None,
                 load_graph: Optional[str] = None,
                 save_graph: Optional[str] = None) -> None:

        super().__init__()

        if load_graph is not None:
            filename = f"stage_{stage}.gpickle"
            filepath = os.path.join(load_graph, filename)
            graph = g.load_graph(filepath)
        else:
            graph = g.build_graph(nodes, k, p, seed=(stage*seed))

            if save_graph is not None:
                directory = f'nodes_{nodes}_seed_{seed}'
                directory_path = os.path.join(save_graph, directory)

                if not os.path.isdir(directory_path):
                    os.makedirs(directory_path, exist_ok=True)

                filename = f'stage_{stage}.gpickle'
                filepath = os.path.join(directory_path, filename)

                g.save_graph(graph, filepath)

        self.layers = nn.Sequential(
            GraphBlock(graph, in_channels, out_channels, kernel_size),
            EfficientAttention(out_channels, out_channels,
                               out_channels, heads))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
