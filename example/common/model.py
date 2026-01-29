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

import dataclasses
from typing import Literal

import torch
from torch import nn

import torch_geometric as pyg

from floydnet import PivotalAttentionBlock

from common.graph import graph_preprocess

@dataclasses.dataclass
class ModelConfig:
    n_embd: int = 768
    n_head: int = 12
    n_out: int = 1
    depth: int = 12
    enable_adj_emb: bool = True
    enable_diffusion: bool = False
    task_level: str = "g"
    n_edge_feat: int = 0
    edge_feat_vocab_size: int = -1
    n_decode_layers: int = 1
    decoder_mask_by_adj: bool = False
    enable_ffn: bool = True
    node_feat_vocab_size: int = 0
    n_node_feat: int = 0
    supernode: bool = False
    dropout: float = 0.0
    norm_fn: str = "affine"

class DiffusionEmbedder(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.linear_t = nn.Linear(n_embd, n_embd, bias=False)
        self.linear_xt = nn.Linear(1, n_embd)

    def init_weights(self):
        nn.init.constant_(self.linear_t.weight, 0.0)
        nn.init.constant_(self.linear_xt.weight, 1)
        nn.init.constant_(self.linear_xt.bias, 0.0)

    def create_time_embedding(self, t, device):
        freqs = torch.arange(self.n_embd // 2, device=device, dtype=torch.float32, requires_grad=False)
        alpha = torch.clamp(t, 0.0, 1.0) + 1e-6
        beta = torch.clamp(1.0 - t, 0.0, 1.0) + 1e-6
        log_snr = torch.log10(alpha / beta)
        scaled_log_snr = log_snr / 60
        time_emb = torch.cat((torch.sin(scaled_log_snr[..., None, None, None] * freqs),
                                torch.cos(scaled_log_snr[..., None, None, None] * freqs)), dim=-1)
        return time_emb

    def forward(self, graph):
        time_emb = self.create_time_embedding(graph.alpha_bar_t, device=graph.xt.device)

        z = self.linear_t(time_emb)
        z = z + self.linear_xt(graph.xt.unsqueeze(-1))
        return z

class FFN(nn.Module):
    def __init__(self, n_embd: int, n_out: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        if n_layers == 1:
            self.net = nn.Linear(n_embd, n_out, bias=True)
        else:
            layers = [
                nn.Linear(n_embd, n_embd * 4, bias=True),
                nn.SiLU(),
            ]
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(n_embd * 4, n_embd * 4, bias=True))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(n_embd * 4, n_out, bias=True))
            self.net = nn.Sequential(*layers)

    def init_weights(self):
        if self.n_layers == 1:
            nn.init.eye_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        else:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        return self.net(x)


class FloydNet(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config

        if config.enable_adj_emb:
            self.emb_adj = nn.Embedding(2, config.n_embd)
        if config.n_edge_feat > 0 and config.edge_feat_vocab_size > 0:
            if config.n_edge_feat > 1:
                self.emb_edge = nn.ModuleList([nn.Embedding(config.edge_feat_vocab_size, config.n_embd) for _ in range(config.n_edge_feat)])
            else: 
                self.emb_edge = nn.Embedding(config.edge_feat_vocab_size, config.n_embd)
        if config.enable_diffusion:
            self.diffusion_embedder = DiffusionEmbedder(config.n_embd)
        if config.n_node_feat > 0 and config.node_feat_vocab_size > 0:
            self.emb_node_i = nn.ModuleList([nn.Embedding(config.node_feat_vocab_size, config.n_embd) for _ in range(config.n_node_feat)])
            self.emb_node_j = nn.ModuleList([nn.Embedding(config.node_feat_vocab_size, config.n_embd) for _ in range(config.n_node_feat)])
        if config.supernode:
            self.emb_superedge = nn.Embedding(4, config.n_embd)

        self.blocks = nn.ModuleList([PivotalAttentionBlock(embed_dim=config.n_embd, num_heads=config.n_head, activation_fn="silu", norm_fn=self.config.norm_fn, enable_ffn=config.enable_ffn, dropout=self.config.dropout)  for _ in range(config.depth)])
        if "g" in config.task_level:
            self.head_g = FFN(config.n_embd, config.n_out, config.n_decode_layers)
        if "v" in config.task_level:
            self.head_v = FFN(config.n_embd, config.n_out, config.n_decode_layers)
        if "e" in config.task_level:
            self.head_e = FFN(config.n_embd, config.n_out, config.n_decode_layers)

    def init_weights(self):
        if hasattr(self, "emb_adj"):
            nn.init.normal_(self.emb_adj.weight, mean=0.0, std=1.0)
        if hasattr(self, "emb_edge"):
            if self.config.n_edge_feat > 1:
                for emb in self.emb_edge:
                    nn.init.normal_(emb.weight, mean=0.0, std=1.0)
            else:
                nn.init.normal_(self.emb_edge.weight, mean=0.0, std=1.0)
        if hasattr(self, "emb_node_i"):
            for emb in self.emb_node_i:
                nn.init.normal_(emb.weight, mean=0.0, std=1.0)
            for emb in self.emb_node_j:
                nn.init.normal_(emb.weight, mean=0.0, std=1.0)
        if hasattr(self, "emb_superedge"):
            nn.init.normal_(self.emb_superedge.weight, mean=0.0, std=1.0)
        if hasattr(self, "diffusion_embedder"):
            self.diffusion_embedder.init_weights()
        if hasattr(self, "head_g"):
            self.head_g.init_weights()
        if hasattr(self, "head_v"):
            self.head_v.init_weights()
        if hasattr(self, "head_e"):
            self.head_e.init_weights()

        for b in self.blocks:
            b._reset_parameters()

    def preprocess(self, graph: pyg.data.Data):
        return graph_preprocess(graph, supernode=self.config.supernode)

    def embed(self, graph: pyg.data.Data):
        x = 0.0
        if self.config.enable_adj_emb:
            x = x + self.emb_adj(graph.adj)
        if self.config.edge_feat_vocab_size > 0:
            if self.config.n_edge_feat > 1:
                for idx in range(self.config.n_edge_feat):
                    x = x + self.emb_edge[idx](graph.adj_attr[:, :, :, idx])
            else:
                x = x + self.emb_edge(graph.adj_attr[:, :, :, 0])
        if self.config.n_node_feat > 0 and self.config.node_feat_vocab_size > 0:
            if self.config.supernode:
                graph.x = graph.x.to(torch.long)
                emb_node_i = 0
                emb_node_j = 0
                for idx in range(self.config.n_node_feat):
                    emb_node_i = emb_node_i + self.emb_node_i[idx](graph.x[:, 1:, idx])
                    emb_node_j = emb_node_j + self.emb_node_j[idx](graph.x[:, 1:, idx])
                # emb_node_i & j: [B, N - 1, c]
                # add to superedge, which is first row and first column
                # take care supernode it self is removed
                n = graph.x.shape[1]
                emb_node_i = emb_node_i[:, :, None]
                emb_node_j = emb_node_j[:, None, :]
                # i: [B, N - 1, 1, c] -> [B, N, N, c]
                emb_node_i = torch.nn.functional.pad(emb_node_i, (0, 0, 0, n - 1, 1, 0))
                # j: [B, 1, N - 1, c] -> [B, N, N, c]
                emb_node_j = torch.nn.functional.pad(emb_node_j, (0, 0, 1, 0, 0, n - 1))

                x = x + emb_node_i
                x = x + emb_node_j
            else:
                raise ValueError("Supernode must be enabled when using node features.")
        if self.config.enable_diffusion:
            x = x + self.diffusion_embedder(graph)
        if self.config.supernode:
            x = x + self.emb_superedge(graph.adj_superedge)
        
        x = x * graph.pair_mask.unsqueeze(-1)
        return x

    def forward(self, graph: pyg.data.Data):
        # [B, N, N], 0 means padding position
        pair_mask = graph.pair_mask
        # [B, 1, N, 1, N], True positions will be filled to -inf
        attn_mask = ~(pair_mask[:, None, :, None, :].bool())
        assert pair_mask.dtype in (torch.float32, torch.bfloat16) and torch.all((pair_mask == 0) | (pair_mask == 1))

        with torch.amp.autocast('cuda', enabled=False):
            x = self.embed(graph)
        for b in self.blocks:
            x = b(x, attn_mask)
            x = x * pair_mask.unsqueeze(-1)
        if self.config.decoder_mask_by_adj:
            x = x * graph.adj.unsqueeze(-1)
        pred_g = pred_v = pred_e = None
        if "g" in self.config.task_level:
            g = x.sum(dim=(1, 2))
            pred_g = self.head_g(g)
        if "v" in self.config.task_level:
            v = x.sum(dim=1) + x.sum(dim=2)
            pred_v = self.head_v(v)
        if "e" in self.config.task_level:
            e = x
            pred_e = self.head_e(e)

        return pred_g, pred_v, pred_e