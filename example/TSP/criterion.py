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

class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, graph):
        pred = pred[-1]
        targets = graph.adj_label
        loss = (targets - pred.squeeze(-1)) ** 2
        loss = loss * graph.pair_mask
        loss = loss.sum(dim=(1, 2)) / graph.pair_mask.sum(dim=(1, 2)).clamp(min=1)
        loss = loss.mean()
        n = targets.shape[1]
        loss = loss * n
        return loss, {}