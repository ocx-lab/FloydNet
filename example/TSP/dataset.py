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

import os
import random
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from torch_geometric import data

from common.utils import is_master_process, print0

class TSPDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        unique: bool,
        split: str,
        subset: str,
        min_n=20,
        max_n=100,
        transform=None,
        split_factor=100,
    ):
        super().__init__(None, transform, None, None)
        assert subset in ("euc", "exp"), f"Unknown subset {subset}"
        self.subset = subset
        self.root = Path(root) / subset / ("uni" if unique else "non-uni") / "raw"

        self.init_samples(min_n, max_n, split, split_factor)

    def init_samples(self, min_n, max_n, split, split_factor):
        root = self.root
        self.length = 0
        self.data = []
        for N in tqdm(range(min_n, max_n + 1), desc="Loading TSP data", disable=not is_master_process()):
            array_data = np.load(root / f"{N:03d}.npy")
            s = array_data.shape[0]
            if split == "train":
                l, r = 0, s - s // split_factor
            else:
                l, r = s - s // split_factor, s

            if l >= r:
                print0(f"Warning: N={N} has no {split} data")
                continue

            cur_data = []
            for i in range(l, r):
                coords = array_data[i].astype(np.float32)
                gid = f"{N:03d}_{i:05d}"
                cur_data.append((coords, gid))
                self.length += 1
            self.data.extend(cur_data)

        print0(f"Loaded {self.length} {split} samples from [{min_n}, {max_n}], subset={self.subset}, unique={self.root}]")
        
    def get_dmat(self, pos):
        if self.subset == "euc":
            assert pos.shape[1] == 2
            dmat = torch.norm(pos[:, None] - pos[None, :], p=2, dim=-1)
            dmat = torch.round(dmat)
        elif self.subset == "exp":
            assert pos.shape[0] == pos.shape[1]
            dmat = pos

        return dmat.to(torch.long)

    def get_edge_attr_and_label(self, pos):
        dmat_tensor = self.get_dmat(pos)
        n = dmat_tensor.shape[0]
        e_idx = []
        e_attr = []
        y = []
        gt_len = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    e_idx.append((i, j))
                    e_attr.append(dmat_tensor[i, j].item())
                    y.append(1 if abs(i - j) in (1, n - 1) else 0)
                    if abs(i - j) in (1, n - 1):
                        gt_len += dmat_tensor[i, j].item()
        gt_len = gt_len // 2  # each edge counted twice
        return e_idx, e_attr, y, gt_len

    def process_pos(self, pos, gid):
        pos = torch.tensor(pos, dtype=torch.float32)
        e_idx, e_attr, y, gt_len = self.get_edge_attr_and_label(pos)
        edge_index = torch.tensor(e_idx, dtype=torch.long).T
        edge_attr = torch.tensor(e_attr, dtype=torch.long)[:, None]
        y = torch.tensor(y, dtype=torch.long)

        graph = data.Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            gt_len=torch.tensor(gt_len, dtype=torch.long),
            gid=gid,
        )
        return graph

    def get(self, idx):
        return self.process_pos(*self.data[idx])

    def len(self):
        return self.length

    @property
    def processed_file_names(self):
        return []

def transform(graph):
    n = graph.pos.shape[0]
    # add a x field to make our data pipeline happy
    graph.x = torch.ones((n, 1))

    # create edge label
    graph.edge_label_index = graph.edge_index.clone()
    graph.edge_label = graph.y

    ans_edge_indices = torch.where(graph.y == 1)[0]
    graph.ans_edge_index = graph.edge_index[:, ans_edge_indices]
    graph.ans_edge_length = graph.edge_attr[ans_edge_indices, 0]
    graph.ans_total_length = graph.ans_edge_length.sum() / 2

    graph.y = None

    return graph