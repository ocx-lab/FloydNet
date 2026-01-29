from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from common.graph import graph_preprocess

try:
    from ogb.graphproppred.mol_encoder import AtomEncoder
except ImportError as e:
    AtomEncoder = None



@dataclass
class GCNContactConfig:
    layers_mp: int = 5
    layers_post_mp: int = 1
    dim_inner: int = 275
    dropout: float = 0.0
    batchnorm: bool = True
    act: str = "relu"
    agg: str = "mean" 
    edge_decoding: str = "dot"
    gcn_add_self_loops: bool = True
    gcn_normalize: bool = True


class MLPNoAct(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        assert num_layers >= 1
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, dim, bias=True))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GCNContactModel(nn.Module):
    def __init__(self, cfg: Optional[GCNContactConfig] = None):
        super().__init__()
        self.cfg = cfg or GCNContactConfig()
        dim = self.cfg.dim_inner

        if AtomEncoder is None:
            raise ImportError(
                "ogb is required to match LRGB Atom encoder. "
                "Please `pip install ogb` in your environment."
            )
        self.node_encoder = AtomEncoder(emb_dim=dim)

        # Message passing stack
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.cfg.layers_mp):
            self.convs.append(
                GCNConv(
                    in_channels=dim,
                    out_channels=dim,
                    improved=False,
                    cached=False,
                    add_self_loops=self.cfg.gcn_add_self_loops,
                    normalize=self.cfg.gcn_normalize,
                    bias=True,
                )
            )
            if self.cfg.batchnorm:
                self.bns.append(nn.BatchNorm1d(dim))

        self.post_mp = MLPNoAct(dim=dim, num_layers=self.cfg.layers_post_mp)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.node_encoder, "reset_parameters"):
            self.node_encoder.reset_parameters()

        for i, conv in enumerate(self.convs):
            conv.reset_parameters()
            if self.cfg.batchnorm:
                self.bns[i].reset_parameters()

        self.post_mp.reset_parameters()

    def _encode_nodes(self, data: Data) -> torch.Tensor:
        if data.x is None:
            raise ValueError("data.x is required for PCQM4Mv2Contact AtomEncoder.")
        if data.x.dtype != torch.long:
            raise TypeError(f"Expected data.x dtype torch.long, got {data.x.dtype}")
        return self.node_encoder(data.x)

    def _mp(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if self.cfg.batchnorm:
                h = self.bns[i](h)
            h = F.relu(h)
            if self.cfg.dropout > 0:
                h = F.dropout(h, p=self.cfg.dropout, training=self.training)
        return h

    def _decode_edges_dot(self, h: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # edge_label_index: [2, E_pred]
        src, dst = edge_label_index[0], edge_label_index[1]
        # single logit per edge (matches dot decoding head expectation)
        return (h[src] * h[dst]).sum(dim=-1, keepdim=True)  # [E_pred, 1]

    def preprocess(self, data):
        return data

    def forward(self, data: Data) -> Tuple[None, None, torch.Tensor]:
        if data.edge_index is None:
            raise ValueError("data.edge_index is required.")
        if not hasattr(data, "edge_label_index") or data.edge_label_index is None:
            raise ValueError("data.edge_label_index [2, E_pred] is required for edge prediction.")

        h = self._encode_nodes(data)
        h = self._mp(h, data.edge_index)

        # Head post-mp on nodes, then dot decode
        h = self.post_mp(h)

        data.x = h
        data = graph_preprocess(data, supernode=False)
        logits = data.x @ data.x.transpose(1, 2)
        logits = logits.unsqueeze(-1)  # [B, N, N, 1]
        return None, None, logits
        # original impl with edge decoding:
        # logits = self._decode_edges_dot(h, data.edge_label_index)
        # data.x = h
        # return None, None, logits
