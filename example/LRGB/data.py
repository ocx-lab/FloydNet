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

def transform_pcqm(graph):
    # format Data(x=[36, 9], edge_index=[2, 72], edge_attr=[72, 3], edge_label_index=[2, 84], edge_label=[84])
    # add supernode at the beginning
    graph.x = torch.nn.functional.pad(graph.x, (0, 0, 1, 0))
    graph.edge_index = graph.edge_index + 1
    graph.edge_label_index = graph.edge_label_index + 1
    return graph