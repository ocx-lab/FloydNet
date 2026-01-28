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
import json
import argparse
from pathlib import Path
from functools import partial

import wandb
import torch

from common.utils import compute_init, print0, compute_cleanup, is_ddp_initialized, DummyWandb, is_master_process, load_checkpoint
from common.dataloader import create_dataloader
from common.model import ModelConfig, FloydNet
from common.trainer import Trainer
from common.diffusion_model import DDPM
from TSP.dataset import TSPDataset, transform
from TSP.criterion import MSE


def parse_args():
    parser = argparse.ArgumentParser(description="Run FloydNet for TSP task")
    # data
    parser.add_argument("--data_dir", type=str, default="data/TSP", help="Directory for the dataset")
    parser.add_argument("--subset", type=str, choices=["euc", "exp"], default="euc", help="TSP dataset subset, euc for Metric TSP, exp for Non-Metric TSP")
    parser.add_argument("--unique", action=argparse.BooleanOptionalAction, default=False, help="Whether to use TSP instances with unique solution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--split_factor", type=int, default=100, help="Dataset split factor for train/val/test split. Val/test will use 1/split_factor of the data.")
    # model
    parser.add_argument("--seed", type=int, default=158293, help="Random seed for initialization")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=96, help="Number of floyd transformer layers")
    # training
    parser.add_argument("--max_train_epoch_len", type=int, default=100, help="Maximum length of training epoch in number of samples")
    parser.add_argument("--max_val_epoch_len", type=int, default=20, help="Maximum length of validation epoch in number of samples")
    parser.add_argument("--max_test_epoch_len", type=int, default=20, help="Maximum length of test epoch in number of samples")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.9), help="Betas for AdamW optimizer")
    parser.add_argument("--grad_clip", type=float, default=100.0, help="Gradient clipping value")
    parser.add_argument("--grad_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval (in epochs)")
    parser.add_argument("--test_interval", type=int, default=200, help="Test interval (in epochs)")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval (in epochs)")
    # output
    parser.add_argument("--wandb_name", type=str, default="dummy", help="WandB experiment name, dummy to disable WandB logging")
    parser.add_argument("--output_dir", type=str, default="outputs/TSP", help="Directory to save outputs")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to load checkpoint from")
    # test
    parser.add_argument("--test_mode", action=argparse.BooleanOptionalAction, default=False, help="Whether to run in test mode (load checkpoint and run test only)")
    parser.add_argument("--sample_count_per_case", type=int, default=10, help="Number of samples to generate per test case during testing")
    args = parser.parse_args()
    return args


def build_dataloader(args):
    dataset_builder = partial(
        TSPDataset,
        root=args.data_dir,
        unique=args.unique,
        subset=args.subset,
        transform=partial(transform),
    )
    if args.test_mode:
        print0("Test mode enabled, loading test dataloader only")
        dataloaders = create_dataloader(
            dataset=dataset_builder,
            batch_size=args.batch_size,
            train=False,
            data_config=dict(
                test=dict(min_n=101, max_n=200, split_factor=args.split_factor)
            )
        )
    else:
        dataloaders = create_dataloader(
            dataset=dataset_builder,
            batch_size=args.batch_size,
            train=True,
            data_config=dict(
                train=dict(min_n=20, max_n=100, split_factor=args.split_factor),
                val=dict(min_n=20, max_n=100, split_factor=args.split_factor),
                # limit to 160 for faster training, we will do full test later
                test=dict(min_n=101, max_n=160, split_factor=args.split_factor),
            )
        )
    return dataloaders


def build_model(args):
    if args.load_checkpoint is not None:
        print0(f"Loading model from checkpoint: {args.load_checkpoint}")
        ckpt_model_config_path = Path(args.load_checkpoint).parent / "model_config.json"
        print0(f"Using model config from checkpoint, ignoring command line model config args")
        with open(ckpt_model_config_path, "r") as f:
            model_config = json.load(f)
    else:
        model_config = dict(
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_out=1,
            depth=args.depth,
            enable_adj_emb=False,
            enable_diffusion=True,
            n_edge_feat=1,
            edge_feat_vocab_size=220,
            task_level="e",
            n_decode_layers=1,
        )
    
    start_epoch = 0
    model = FloydNet(ModelConfig(**model_config)).to("cuda")
    model.init_weights()
    print0(model)
    if is_ddp_initialized():
        torch.distributed.barrier()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

    if args.test_mode:
        optimizer = scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.adam_betas), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)

    if args.load_checkpoint is not None:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            checkpoint_path=args.load_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print0(f"Resumed from epoch {start_epoch}")
            
    if is_master_process() and not args.test_mode:
        # save model config for loading model later
        output_dir = Path(args.output_dir) / "checkpoints"
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=4)

    return model, optimizer, scheduler, model_config, start_epoch


def main(args):
    print0("Initializing distributed environment...")
    compute_init(seed=args.seed)

    print0("Building dataloaders...")
    dataloaders = build_dataloader(args)
    print0("Building model, optimizer, and scheduler...")
    model, optimizer, scheduler, model_config, start_epoch = build_model(args)

    diffusion_model = DDPM().cuda()
    criterion = MSE()
    logger = DummyWandb() if (args.wandb_name == "dummy" or not is_master_process() or args.test_mode) else wandb.init(
        project="floydnetwork",
        name=args.wandb_name,
        config=vars(args).copy(),
    )
    if args.test_mode:
        print0("max test epoch len is set to large enough for testing")
        args.max_test_epoch_len = 1e9
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        critn=criterion,
        out_dir=args.output_dir,
        max_epochs=args.max_epochs,
        amp=True,
        grad_clip_val=args.grad_clip,
        grad_accumulation=args.grad_accumulation,
        eval_interval=args.eval_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        diffusion_model=diffusion_model,
        max_train_epoch_len=args.max_train_epoch_len,
        max_val_epoch_len=args.max_val_epoch_len,
        max_test_epoch_len=args.max_test_epoch_len,
        eval_fn="TSP",
    )
    if args.test_mode:
        print0("Starting testing...")
        trainer.test(dataloaders, sample_count_per_case=args.sample_count_per_case, epoch=start_epoch)
    else:
        print0("Starting training...")
        trainer.fit(logger, dataloaders, start_epoch=start_epoch)

    compute_cleanup()
    print0("Finished all processes.")


if __name__ == '__main__':
    args = parse_args()
    main(args)