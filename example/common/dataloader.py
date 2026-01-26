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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch_geometric import loader

from common.utils import print0, is_ddp_initialized

def create_dataloader(
    dataset,
    batch_size: int,
    train: bool,
    data_config: dict = {},
):
    dataloaders_dict = {}
    if not train:
        splits = ["test"]
    else:
        splits = ["train", "val", "test"]
    
    for split_name in splits:
        cur_dataset = dataset(split=split_name, **data_config.get(split_name, {}))

        print0(f"Creating dataloader for {split_name} split with {len(cur_dataset)} samples")

        if is_ddp_initialized():
            sampler = DistributedSampler(cur_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        else:
            sampler = None

        dataloaders_dict[split_name] = loader.DataLoader(
            cur_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=(sampler is None),
            follow_batch=["y", "y_node", "y_edge"], 
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True,
        )
    return dataloaders_dict