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

from typing import Optional
import math

import torch
import torch.nn.functional as F

def pivotal_attention(
    q_ik: torch.Tensor,
    k_ij: torch.Tensor,
    k_jk: torch.Tensor,
    v_ij: torch.Tensor,
    v_jk: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    inf: float = 1e9,
) -> torch.Tensor:
    """Pivotal attention as described in "FLOYDNET: A LEARNING PARADIGM FOR GLOBAL RELATIONAL REASONING".

    Shapes:
        q_ik: (B, H, L_i, L_k, D)
        k_ij: (B, H, L_i, L_j, D)
        k_jk: (B, H, L_j, L_k, D)
        v_ij: (B, H, L_i, L_j, D)
        v_jk: (B, H, L_j, L_k, D)
        attn_mask (optional): broadcastable to (B, H, L_i, L_k, L_j)

    Args:
        attn_mask: Additive mask (float) or boolean mask. If boolean, masked positions are set to -inf.
        dropout: Dropout probability applied to attention weights (only effective if > 0).
        scale: Optional custom scaling factor. If None, defaults to 1/sqrt(2*D).
        inf: Value to use for -infinity in masks.

    Returns:
        Tensor of shape (B, H, L_i, L_k, D)
    """
    assert all([t.dim() == 5 for t in [q_ik, k_ij, k_jk, v_ij, v_jk]]), "All inputs must be 5D tensors"
    B, H, L_i, L_k, D = q_ik.shape
    L_j = k_ij.shape[3]
    assert k_ij.shape == v_ij.shape == (B, H, L_i, L_j, D), "k_ij and v_ij must have shape (B, H, L_i, L_j, D)"
    assert k_jk.shape == v_jk.shape == (B, H, L_j, L_k, D), "k_jk and v_jk must have shape (B, H, L_j, L_k, D)"

    if scale is None:
        scale = 1.0 / math.sqrt(2.0 * D)
    q_ik = q_ik * scale

    # Compute attention scores over the pivot dimension j: (B, H, L_i, L_k, L_j)
    attn_scores = torch.einsum("bhikd,bhijd->bhikj", q_ik, k_ij) \
                + torch.einsum("bhikd,bhjkd->bhikj", q_ik, k_jk)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(attn_mask, -inf)
        else:
            attn_scores = attn_scores + attn_mask

    attn_weights = torch.softmax(attn_scores, dim=-1)

    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout)

    y = torch.einsum("bhikj,bhijd->bhikd", attn_weights, v_ij) \
      + torch.einsum("bhikj,bhjkd->bhikd", attn_weights, v_jk)

    return y

def pivotal_attention3(
    q_ijk: torch.Tensor,
    k_pjk: torch.Tensor,
    k_ipk: torch.Tensor,
    k_ijp: torch.Tensor,
    v_pjk: torch.Tensor,
    v_ipk: torch.Tensor,
    v_ijp: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    inf: float = 1e9,
) -> torch.Tensor:
    """3-Pivotal attention as described in "FLOYDNET: A LEARNING PARADIGM FOR GLOBAL RELATIONAL REASONING".

    Shapes:
        q_ijk: (B, H, L_i, L_j, L_k, D)
        k_pjk: (B, H, L_p, L_j, L_k, D)
        k_ipk: (B, H, L_i, L_p, L_k, D)
        k_ijp: (B, H, L_i, L_j, L_p, D)
        v_pjk: (B, H, L_p, L_j, L_k, D)
        v_ipk: (B, H, L_i, L_p, L_k, D)
        v_ijp: (B, H, L_i, L_j, L_p, D)
        attn_mask (optional): broadcastable to (B, H, L_i, L_j, L_k, L_p)

    Args:
        attn_mask: Additive mask (float) or boolean mask. If boolean, masked positions are set to -inf.
        dropout: Dropout probability applied to attention weights (only effective if > 0).
        scale: Optional custom scaling factor. If None, defaults to 1/sqrt(3*D).
        inf: Value to use for -infinity in masks.

    Returns:
        Tensor of shape (B, H, L_i, l_j, L_k, D)
    """
    assert all([t.dim() == 6 for t in [q_ijk, k_pjk, k_ipk, k_ijp, v_pjk, v_ipk, v_ijp]]), "All inputs must be 6D tensors"
    B, H, L_i, L_j, L_k, D = q_ijk.shape
    L_p = k_pjk.shape[2]
    assert k_pjk.shape == v_pjk.shape == (B, H, L_p, L_j, L_k, D), "k_pjk and v_pjk must have shape (B, H, L_p, L_j, L_k, D)"
    assert k_ipk.shape == v_ipk.shape == (B, H, L_i, L_p, L_k, D), "k_ipk and v_ipk must have shape (B, H, L_i, L_p, L_k, D)"
    assert k_ijp.shape == v_ijp.shape == (B, H, L_i, L_j, L_p, D), "k_ijp and v_ijp must have shape (B, H, L_i, L_j, L_p, D)"
    
    if scale is None:
        scale = 1.0 / math.sqrt(3.0 * D)
    q_ijk = q_ijk * scale

    # Compute attention scores over the pivot dimension j: (B, H, L_i, L_j, L_k, L_p)
    attn_scores = torch.einsum("bhijkd,bhpjkd->bhijkp", q_ijk, k_pjk) \
                + torch.einsum("bhijkd,bhipkd->bhijkp", q_ijk, k_ipk) \
                + torch.einsum("bhijkd,bhijpd->bhijkp", q_ijk, k_ijp)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(attn_mask, -inf)
        else:
            attn_scores = attn_scores + attn_mask

    attn_weights = torch.softmax(attn_scores, dim=-1)

    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout)

    y = torch.einsum("bhijkp,bhpjkd->bhijkd", attn_weights, v_pjk) \
      + torch.einsum("bhijkp,bhipkd->bhijkd", attn_weights, v_ipk) \
      + torch.einsum("bhijkp,bhijpd->bhijkd", attn_weights, v_ijp)

    return y

