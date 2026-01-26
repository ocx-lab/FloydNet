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

from __future__ import annotations

import copy
from typing import Optional, Tuple, Union, Callable

import torch
from torch import nn
from .functional import pivotal_attention


class Affine(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((c, )))
        self.bias = nn.Parameter(torch.zeros((c, )))

    def forward(self, x: torch.Tensor):
        return x * self.weight + self.bias


def create_norm(norm_fn: Union[str, Callable], embed_dim: int, eps: float = 1e-5, **kwargs) -> nn.Module:
    """Create a normalization module from a name or nn.Module.

    Args:
        norm_fn: Name or an nn.Module instance/class.
        embed_dim: Embedding dimension (features) used to construct the norm.
        eps: Numerical epsilon passed to the normalization layer if applicable.
        **kwargs: Extra keyword arguments forwarded to the normalization layer.

    Returns:
        An nn.Module normalization instance.
    """
    if isinstance(norm_fn, str):
        if norm_fn.lower() in ["layernorm", "ln"]:
            return nn.LayerNorm(embed_dim, eps=eps, **kwargs)
        elif norm_fn.lower() in ["batchnorm", "bn"]:
            return nn.BatchNorm1d(embed_dim, eps=eps, **kwargs)
        elif norm_fn.lower() in ["rmsnorm", "rms"]:
            return nn.RMSNorm(embed_dim, eps=eps, **kwargs)
        elif norm_fn.lower() in ["affine"]:
            return Affine(embed_dim)
        elif norm_fn.lower() in ["none", "identity"]:
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_fn string: {norm_fn}")
    elif callable(norm_fn):
        if isinstance(norm_fn, nn.Module):
            # deepcopy to avoid shared parameters
            return copy.deepcopy(norm_fn)
        elif isinstance(norm_fn, type) and issubclass(norm_fn, nn.Module):
            return norm_fn(embed_dim, eps=eps, **kwargs)
        else:
            raise TypeError("norm_fn callable must be an nn.Module or nn.Module class")
    else:
        raise TypeError("norm_fn must be a string or callable")

        
def create_activation(activation_fn: Union[str, Callable]) -> nn.Module:
    """Create an activation module from a name or nn.Module.

    Args:
        activation_fn: Name or an nn.Module instance/class.

    Returns:
        An nn.Module activation instance.
    """
    if isinstance(activation_fn, str):
        if activation_fn.lower() == "relu":
            return nn.ReLU()
        elif activation_fn.lower() == "gelu":
            return nn.GELU()
        elif activation_fn.lower() == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation_fn string: {activation_fn}")
    elif callable(activation_fn):
        if isinstance(activation_fn, nn.Module):
            return activation_fn
        elif isinstance(activation_fn, type) and issubclass(activation_fn, nn.Module):
            return activation_fn()
        else:
            raise TypeError("activation_fn callable must be an nn.Module or nn.Module class")
    else:
        raise TypeError("activation_fn must be a string or callable")


class PivotalAttentionBlock(nn.Module):
    """Transformer-style block that applies pivotal attention followed by an FFN.

    Args:
        embed_dim: Input/hidden embedding dimension (D).
        num_heads: Number of attention heads (D must be divisible by num_heads).
        dropout: Dropout probability for attention output and FFN output.
        bias: Whether to include bias terms in linear layers.
        ffn_expansion_ratio: Expansion ratio for the FFN hidden size.
        norm_position: "pre" or "post" layer normalization placement.
        activation_fn: Activation name/module used in the FFN.
        norm_fn: Normalization name/module used in the block.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        ffn_expansion_ratio: int = 4,
        norm_position: str = "pre",
        activation_fn: Union[str, Callable] = "relu",
        norm_fn: Union[str, Callable] = "layernorm",
        enable_symmetric_mix: bool = True,
        enable_ffn: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.norm_position = norm_position.lower()
        self.enable_ffn = enable_ffn
        assert self.norm_position in ["pre", "post"], "norm_position must be 'pre' or 'post'"

        self.enable_symmetric_mix = enable_symmetric_mix
        if enable_symmetric_mix:
            self.c_mix = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.c_qkv = nn.Linear(embed_dim, embed_dim * 5, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_fn = nn.Dropout(dropout)
        self.norm1 = create_norm(norm_fn, embed_dim)
        if self.enable_ffn:
            self.activation_fn = create_activation(activation_fn)
            self.norm2 = create_norm(norm_fn, embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_expansion_ratio * embed_dim, bias=bias),
                self.activation_fn,
                nn.Linear(ffn_expansion_ratio * embed_dim, embed_dim, bias=bias),
                nn.Dropout(dropout),
            )
            self.ffn_scale = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier for projections and zeros for output heads."""
        if self.enable_symmetric_mix:
            nn.init.zeros_(self.c_mix.weight)
        nn.init.xavier_uniform_(self.c_qkv.weight)
        nn.init.zeros_(self.c_proj.weight)
        if self.enable_ffn:
            nn.init.xavier_uniform_(self.ffn[0].weight)
            nn.init.zeros_(self.ffn[2].weight)
        if self.c_qkv.bias is not None:
            if self.enable_symmetric_mix:
                nn.init.zeros_(self.c_mix.bias)
            nn.init.zeros_(self.c_qkv.bias)
            nn.init.zeros_(self.c_proj.bias)
            if self.enable_ffn:
                nn.init.zeros_(self.ffn[0].bias)
                nn.init.zeros_(self.ffn[2].bias)

    def attn(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply pivotal attention over a (L x L) grid.

        Args:
            x: Input tensor of shape (B, L, L, D).
            attn_mask: Optional mask broadcastable to (B, H, L, L, L).

        Returns:
            Tensor of shape (B, L, L, D) after attention projection and dropout.
        """
        B, L, _, D = x.shape
        # [B, L, L, 5*D] -> 5 x [B, H, L, L, d]
        qkv = torch.chunk(self.c_qkv(x), 5, dim=-1)
        q_ik, k_ij, k_jk, v_ij, v_jk = map(
            lambda t: t.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4),
            qkv, 
        )

        # [B, H, L, L, d]
        y = pivotal_attention(
            q_ik, k_ij, k_jk, v_ij, v_jk,
            attn_mask=attn_mask,
            dropout=self.dropout if self.training else 0.0,
        )
        y = y.permute(0, 2, 3, 1, 4).contiguous().view(B, L, L, D)
        y = self.c_proj(y)
        y = self.dropout_fn(y)
        return y

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_symmetric_mix:
            xT = self.c_mix(x.transpose(1, 2))
        else:
            xT = 0
        if self.norm_position == "pre":
            x = x + self.attn(self.norm1(x + xT), attn_mask)
            if self.enable_ffn:
                x = x + self.ffn(self.norm2(x)) * self.ffn_scale
        else:
            x = self.norm1(x + self.attn(x + xT, attn_mask))
            if self.enable_ffn:
                x = self.norm2(x + self.ffn(x)) * self.ffn_scale

        return x