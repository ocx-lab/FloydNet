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
import torch.nn as nn

class GraphCountLoss(nn.Module):
    def __init__(self, task: str):
        super().__init__()
        self.task = task

        self.normalize = {
            "cycle3,v": 3, "cycle3,e": 6, "cycle4,v": 4, "cycle4,e": 8,
            "cycle5,v": 5, "cycle5,e": 10, "cycle6,v": 6, "cycle6,e": 12,
            "chordal4,v": 4, "chordal4,e": 10, "chordal5,v": 5, "chordal5,e": 14,
        }

    def _get_scale(self, level):
        task_identifier = f"{self.task},{level}"
        return 1 / self.normalize.get(task_identifier, 1)
        

    def forward(self, pred, graph):
        pred_g, pred_v, pred_e = pred

        loss_g = torch.abs(pred_g.squeeze(-1) - graph.y_graph) * self._get_scale("g")

        loss_v = torch.abs(pred_v.squeeze(-1) - graph.y_node)
        loss_v = (loss_v * graph.single_mask).sum(dim=1) / graph.single_mask.sum(dim=1).clamp(min=1)
        loss_v = loss_v * self._get_scale("v")

        loss_e = torch.abs(pred_e.squeeze(-1) - graph.y_edge)
        pair_mask = graph.adj * graph.pair_mask  # only consider existing edges
        loss_e = (loss_e * pair_mask).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp(min=1)
        loss_e = loss_e * self._get_scale("e")

        loss_g = loss_g.mean()
        loss_v = loss_v.mean()
        loss_e = loss_e.mean()

        l1_loss = loss_g + loss_v + loss_e

        return l1_loss, {
            "loss_g": loss_g.item(),
            "loss_v": loss_v.item(),
            "loss_e": loss_e.item(),
        }