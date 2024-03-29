# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        in_features = out_features = 100
        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.linear = nn.Linear(in_features, out_features)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x = batched_data["x"].clone(),
        n_graph, n_node,features = batched_data["x"].size()
        # x = torch.tensor(x).to(torch.int64)
        # valid_mask = (x >= 0) & (x < self.atom_encoder.num_embeddings)
        # x = x.masked_fill(~valid_mask, 0)
        # node feauture + graph token
        # x1 = torch.arange(0, 16 * 66*100, dtype=torch.long).view(16, 66,100)
        # x1 = torch.ones(16, 66, 100, dtype=torch.long)
        # x1 = x1.to('cuda:0')
        x_reshaped = batched_data["x"].view(-1, features)
        x_transformed = self.linear(x_reshaped)
        node_feature= x_transformed.view(n_graph, n_node, features)
        # node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        # if self.flag and perturb is not None:
        #     node_feature += perturb
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        # graph_node_feature = node_feature.clone()


        return graph_node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.in_features=66
        self.out_heads=10
        self.fc = nn.Linear(self.in_features, self.out_heads * self.in_features)
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        spatial_pos, x,attn_bias = (
            batched_data["spatial_pos"],
            batched_data["x"],
            batched_data["attn_bias"]
        )
        # spatial_pos=batched_data["spatial_pos"]
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        n_graph, n_node = x.size()[:2]

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]
        spatial_pos=spatial_pos.float()
        spatial_pos_bias = self.fc(spatial_pos)
        spatial_pos_bias = spatial_pos_bias.view(n_graph, n_node, self.out_heads, self.in_features)
        spatial_pos_bias = spatial_pos_bias.permute(0,2,1,3)


        # spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:]
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
        return graph_attn_bias
