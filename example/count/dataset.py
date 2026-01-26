# This file is derived from the following project:
#     https://github.com/subgraph23/homomorphism-expressivity
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

import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as data

class GraphCount(data.InMemoryDataset):

    TASK = [
        "boat", "chordal6", "chordal4_1", "chordal4_4", "chordal5_13", "chordal5_31", "chordal5_24", # HOM
        "cycle3", "cycle4", "cycle5", "cycle6", "chordal4", "chordal5", # ISO
    ]

    def __init__(self, root: str, split: str, task: str, **kwargs):
        super().__init__(root, **kwargs)

        assert task in self.TASK

        graph, i = torch.load(f"{self.processed_dir}/{split}.pt", weights_only=False)
        self.data = data.Data(graph.x, graph.edge_index, y_node=graph[f"{task}:v"], y_edge=graph[f"{task}:e"], y_graph=graph[f"{task}:g"])
        self.slices = dict(x=i["x"], edge_index=i["edge_index"], y_node=i[f"{task}:v"], y_edge=i[f"{task}:e"], y_graph=i[f"{task}:g"])

    @property
    def raw_file_names(self):
        return ["graph.npy", "hom.npy", "iso.npy"]

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def process(self):

        def to_pyg(A, hom, iso):
            to_pt = lambda count: { f"{task}:{level}": torch.from_numpy(count)
              for task, value in count.items() for level, count in zip(["g", "v", "e"], value) }
            return data.Data(
                x=torch.ones(len(A), 1, dtype=torch.int64), **to_pt(hom), **to_pt(iso),
                edge_index=torch.Tensor(np.vstack(np.where(A != 0))).type(torch.int64),
            )

        load = lambda f: np.load(f"{self.root}/{f}", allow_pickle=True)
        (graph, index), *count = map(load, self.raw_file_names)

        graph = map(to_pyg, graph, *count)

        if self.pre_filter is not None:
            graph = filter(self.pre_filter, graph)

        if self.pre_transform is not None:
            graph = map(self.pre_transform, graph)

        graph = list(graph) # run processing

        normalize = { task: torch.std(torch.cat([G[f"{task}:g"]
                      for G in graph])) for task in self.TASK }

        for split in ["train", "val", "test"]:

            from operator import itemgetter as get
            graph_split = get(*index[()][split])(graph)
            graph_split, ix = self.collate(graph_split)

            for key, std in normalize.items():
                graph_split[f"{key}:g"] /= std
                graph_split[f"{key}:v"] /= std
                graph_split[f"{key}:e"] /= std

            torch.save((graph_split, ix), f"{self.processed_dir}/{split}.pt")