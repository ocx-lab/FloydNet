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
from torch import nn
from einops import rearrange

from floydnet import pivotal_attention, pivotal_attention3

from common.graph import graph_preprocess

def norm(x):
    return (x - x.mean()) / x.std()

class FloydBlock2(nn.Module):
    def __init__(self, idx, n_embd, n_head, **kwargs):
        super().__init__()
        head_dim = n_embd // n_head
        self.n_head = n_head
        self.scale = (head_dim ** -0.5) * 50
        self.c_qkv = nn.Linear(n_embd, n_embd * 5, bias=False)

    def forward(self, x):
        x = norm(x)
        b, n, _, c = x.shape  # (b, n, n, c)
        qkv = self.c_qkv(x).chunk(5, dim=-1)
        q, k1, k2, v1, v2 = map(lambda t: rearrange(t, 'b i j (h d) -> b h i j d', h=self.n_head), qkv)
        x = pivotal_attention(q, k1, k2, v1, v2, scale=self.scale)
        x = rearrange(x, 'b h i j d -> b i j (h d)')
        return x

class FloydBlock3(nn.Module):
    def __init__(self, idx, n_embd, n_head, sf, use_norm):
        super().__init__()
        head_dim = n_embd // n_head
        self.n_head = n_head
        self.use_norm = use_norm
        self.scale = (head_dim ** -0.5) * sf
        self.c_qkv = nn.Linear(n_embd, n_embd * 7, bias=False)

    def forward(self, x):
        if self.use_norm:
            x = norm(x)
        b, n, _, _, c = x.shape  # (b, n, n, n, c)
        qkv = self.c_qkv(x).chunk(7, dim=-1)
        q, k1, k2, k3, v1, v2, v3 = map(lambda t: rearrange(t, 'b i j k (h d) -> b h i j k d', h=self.n_head), qkv)
        x = pivotal_attention3(q, k1, k2, k3, v1, v2, v3, scale=self.scale)
        x = rearrange(x, 'b h i j k d -> b i j k (h d)')
        return x

class FloydNet(nn.Module):
    def __init__(self, n_embd, n_head, depth, floyd_level, freeze, **block_kwargs):
        super().__init__()
        self.n_embd = n_embd
        self.floyd_level = floyd_level

        assert floyd_level in [2, 3], "floyd_level must be 2 or 3"
        block_cls = FloydBlock2 if floyd_level == 2 else FloydBlock3

        self.emb = nn.Embedding(128, n_embd)
        self.blocks = nn.ModuleList([block_cls(idx, n_embd, n_head, **block_kwargs) for idx in range(depth)])
        self.head = nn.Linear(n_embd, n_embd, bias=True)

        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False
        else:
            for parameter in self.emb.parameters():
                parameter.requires_grad = False
            for parameter in self.blocks.parameters():
                parameter.requires_grad = False

    def init_weights(self):
        torch.cuda.manual_seed(42)
        nn.init.normal_(self.emb.weight, std=1.0)
        for idx, block in enumerate(self.blocks):
            torch.cuda.manual_seed(idx)
            nn.init.normal_(block.c_qkv.weight, std=self.n_embd ** -0.5)
        torch.cuda.manual_seed(42)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def create_hypergraph(self, x: torch.Tensor) -> torch.Tensor:
        # x is the adjacency matrix of shape (b, n, n)
        b, n = x.shape[:2]
        x_new = torch.zeros([b, n, n, n], device=x.device, dtype=x.dtype)
        x_new *= 2
        x_new += x[:, :, :, None]
        x_new *= 2
        x_new += x[:, :, None, :]
        x_new *= 2
        x_new += x[:, None, :, :]
        x = x.transpose(-2, -1)
        x_new *= 2
        x_new += x[:, :, :, None]
        x_new *= 2
        x_new += x[:, :, None, :]
        x_new *= 2
        x_new += x[:, None, :, :]
        return x_new

    def forward(self, graph):
        graph = graph_preprocess(graph)
        x = graph.adj.long()  # (b, n, n)
        
        with torch.no_grad():
            if self.floyd_level == 3:
                x = self.create_hypergraph(x)  # (b, n, n, n)
            x = self.emb(x)  # (b, n, n, c) or (b, n, n, n, c)
            for block in self.blocks:
                x = block(x)

            dims = tuple(range(1, x.dim() - 1))
            x = x.mean(dim=dims)  # (b, c)
        x = self.head(x)
        return x
