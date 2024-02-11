# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch_geometric

from functools import lru_cache
import pyximport
import torch.distributed as dist
import cython
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

@torch.jit.script

def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    spatial_pos=item.edge_attr

    x= x.t()
    N = x.size(0)
    x = convert_to_single_emb(x)
    # node adj matrix [N, N] bool
    # attn_bias = torch.tensor(attn_bias)
    spatial_pos=torch.tensor(spatial_pos)


    # edge feature here
    # if len(edge_attr.size()) == 1:
    #     edge_attr = edge_attr[:, None]
    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.float64)
    # attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
    #     convert_to_single_emb(edge_attr) + 1
    # )
    #
    # max_dist = np.amax(shortest_path_result)
    # edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    # spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = None
    item.spatial_pos = spatial_pos
    item.in_degree = None
    item.out_degree = None  # for undirected graph
    item.edge_input = None

    return item



