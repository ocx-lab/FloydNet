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

import argparse
from functools import partial

import wandb
import torch

from common.utils import compute_init, print0, compute_cleanup, is_ddp_initialized, DummyWandb, is_master_process
from common.dataloader import create_dataloader
from common.model import ModelConfig, FloydNet
from common.trainer import Trainer
from count.dataset import GraphCount
from count.criterion import GraphCountLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Run FloydNet for counting task")
    # data
    parser.add_argument("--data_dir", type=str, default="data/count", help="Directory for the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--task", type=str, required=True, help="Counting task name")
    # model
    parser.add_argument("--seed", type=int, default=158293, help="Random seed for initialization")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=64, help="Number of floyd transformer layers")
    # training
    parser.add_argument("--max_epochs", type=int, default=8000, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.9), help="Betas for AdamW optimizer")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clipping value")
    parser.add_argument("--eval_interval", type=int, default=20, help="Evaluation interval (in epochs)")
    parser.add_argument("--test_interval", type=int, default=100, help="Test interval (in epochs)")
    parser.add_argument("--save_interval", type=int, default=1500, help="Model saving interval (in epochs)")
    # output
    parser.add_argument("--wandb_name", type=str, default="dummy", help="WandB experiment name, dummy to disable WandB logging")
    parser.add_argument("--output_dir", type=str, default="outputs/count", help="Directory to save outputs")
    args = parser.parse_args()
    return args


def build_dataloader(args):
    dataset_builder = partial(
        GraphCount,
        root=args.data_dir,
        task=args.task,
    )
    dataloaders = create_dataloader(
        dataset=dataset_builder,
        batch_size=args.batch_size,
        train=True,
    )
    return dataloaders


def build_model(args):
    model_config = dict(
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_out=1,
        depth=args.depth,
        enable_adj_emb=True,
        enable_diffusion=False,
        task_level="gve",
        n_decode_layers=4,
        decoder_mask_by_adj=True,
        enable_ffn=False,
    )
    model = FloydNet(ModelConfig(**model_config)).to("cuda")
    model.init_weights()
    print0(model)
    if is_ddp_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.adam_betas), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    return model, optimizer, scheduler

def main(args):
    print0("Initializing distributed environment...")
    compute_init(seed=args.seed)

    print0("Building dataloaders...")
    dataloaders = build_dataloader(args)
    print0("Building model, optimizer, and scheduler...")
    model, optimizer, scheduler = build_model(args)
    criterion = GraphCountLoss(task=args.task)
    logger = DummyWandb() if (args.wandb_name == "dummy" or not is_master_process()) else wandb.init(
        project="floydnetwork",
        name=args.wandb_name,
        config=vars(args).copy(),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        critn=criterion,
        out_dir=args.output_dir,
        max_epochs=args.max_epochs,
        amp=False,
        grad_clip_val=args.grad_clip,
        eval_interval=args.eval_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
    )
    print0("Starting training...")
    trainer.fit(logger, dataloaders, start_epoch=0)

    compute_cleanup()
    print0("Finished all processes.")


if __name__ == '__main__':
    args = parse_args()
    main(args)