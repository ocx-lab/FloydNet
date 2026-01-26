# Copyright 2025 Beijing Academy of Artificial Intelligence (BAAI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn.utils.rnn import pad_sequence
import torch_geometric as pyg


def batch_pad_v(x, ptr, counts, max_m):
    """
    input:
        x: [n, c]
    output:
        x: [b, m, c]
        mask: [b, m]

    """
    groups = [x[ptr[i]:ptr[i+1]] for i in range(ptr.shape[0] - 1)]
    padded_x = pad_sequence(groups, batch_first=True, padding_value=0)
    if padded_x.shape[1] < max_m:
        if padded_x.ndim == 3:
            padded_x = torch.nn.functional.pad(padded_x, (0, 0, 0, max_m - padded_x.shape[1]), value=0)
        elif padded_x.ndim == 2:
            padded_x = torch.nn.functional.pad(padded_x, (0, max_m - padded_x.shape[1]), value=0)
        else:
            raise ValueError(f"Unsupported padded_x ndim {padded_x.ndim}")
    # [b, m]
    mask = (torch.arange(max_m, device=x.device)[None, :] < counts[:, None]).float()
    return padded_x, mask


def batch_pad_e(e, ptr, counts, max_m):
    """
    input:
        e: [n, n, ...]
    output:
        e: [b, m, m, ...]
    """
    sh = (counts.shape[0], max_m, max_m)
    if e.ndim > 2:
        sh += e.shape[2:]
    padded_e = e.new_zeros(sh)
    for i in range(ptr.shape[0] - 1):
        submatrix = e[ptr[i]:ptr[i+1], ptr[i]:ptr[i+1]]
        padded_e[i][:counts[i], :counts[i]] = submatrix
    return padded_e

def batch_pad_e_y(y, ptr, counts, max_m):
    """
    input:
        y: [n]
        ptr: [b]
    output:
        y: [b, m, m]
    """
    sh = (counts.shape[0], max_m, max_m)
    padded_y = y.new_zeros(sh)
    for i in range(ptr.shape[0] - 1):
        submatrix = y[ptr[i]:ptr[i+1]].view(counts[i], counts[i])
        padded_y[i][:counts[i], :counts[i]] = submatrix
    return padded_y


def graph_preprocess(graph):
    """
    Convert a PyG-style batched graph object into dense, per-graph padded tensors.

    Args:
        graph: A PyG batched graph/data object. Expected to contain at least
            `x`, `edge_index`, `batch`, and `ptr`. May optionally contain
            `edge_feat_discrete`, `edge_label`/`edge_label_index`, and `y`.

    Returns:
        The same `graph` object with dense/padded tensors attached.
    """
    if 'processed' in graph and graph.processed:
        return graph
    unique_batch, counts = torch.unique(graph.batch, return_counts=True, sorted=True)
    max_m = counts.max()

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    num_edge_features = graph.num_edge_features

    x, mask = batch_pad_v(graph.x, graph.ptr, counts, max_m)

    adj = pyg.utils.to_dense_adj(graph["edge_index"], max_num_nodes=num_nodes).squeeze(0).to(torch.int32)
    adj = batch_pad_e(adj, graph.ptr, counts, max_m)

    if "edge_attr" in graph:
        f, (src, dst) = graph.edge_attr, graph.edge_index
        t = torch.zeros(num_nodes, num_nodes, f.shape[-1], dtype=f.dtype, device=f.device)
        t[src, dst] = f
        graph.adj_attr = batch_pad_e(t, graph.ptr, counts, max_m)

    if "edge_label" in graph:
        edge_label, (src, dst) = graph.edge_label, graph.edge_label_index
        adj_label = torch.zeros(num_nodes, num_nodes, dtype=edge_label.dtype, device=edge_label.device)
        adj_label[src, dst] = edge_label
        graph.adj_label = batch_pad_e(adj_label, graph.ptr, counts, max_m)

    if "y_node" in graph:
        graph.y_node, _ = batch_pad_v(graph.y_node, graph.ptr, counts, max_m)

    if "y_edge" in graph:
        graph.y_edge = batch_pad_e_y(graph.y_edge, graph.y_edge_ptr, counts, max_m)

    graph.x = x
    graph.single_mask = mask
    graph.pair_mask = mask[:, :, None] * mask[:, None, :]
    graph.adj = adj
    graph.processed = True

    return graph
