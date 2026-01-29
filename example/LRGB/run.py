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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
from torch_geometric.datasets import LRGBDataset

from common.utils import compute_init, print0, compute_cleanup, is_ddp_initialized, DummyWandb, is_master_process, load_checkpoint
from common.dataloader import create_dataloader
from common.model import ModelConfig, FloydNet
from common.trainer import Trainer
from LRGB.data import transform_pcqm
from LRGB.criterion import FocalLoss
from LRGB.mrr import EdgeMRR
from LRGB.gcn_model import GCNContactModel, GCNContactConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run FloydNet for LRGB task")
    # data
    parser.add_argument("--data_dir", type=str, default="data/LRGB", help="Directory for the dataset")
    parser.add_argument("--name", type=str, default="pcqm-contact")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for training")
    # model
    parser.add_argument("--seed", type=int, default=158293, help="Random seed for initialization")
    parser.add_argument("--n_embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=32, help="Number of floyd transformer layers")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--arch", type=str, default="floydnet", choices=["floydnet", "gcn"], help="Model architecture to use")
    # training
    parser.add_argument("--max_train_epoch_len", type=int, default=400, help="Maximum length of training epoch in number of samples")
    parser.add_argument("--max_val_epoch_len", type=int, default=1000, help="Maximum length of validation epoch in number of samples")
    parser.add_argument("--max_test_epoch_len", type=int, default=1000, help="Maximum length of test epoch in number of samples")
    parser.add_argument("--max_epochs", type=int, default=5000, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.98), help="Betas for AdamW optimizer")
    parser.add_argument("--grad_clip", type=float, default=100.0, help="Gradient clipping value")
    parser.add_argument("--grad_accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=20, help="Evaluation interval (in epochs)")
    parser.add_argument("--test_interval", type=int, default=20, help="Test interval (in epochs)")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval (in epochs)")
    # output
    parser.add_argument("--wandb_name", type=str, default="dummy", help="WandB experiment name, dummy to disable WandB logging")
    parser.add_argument("--output_dir", type=str, default="outputs/LRGB", help="Directory to save outputs")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to load checkpoint from")
    args = parser.parse_args()
    return args


def build_dataloader(args):
    dataset_builder = partial(
        LRGBDataset,
        root=args.data_dir,
        name=args.name,
        transform=partial(transform_pcqm),
    )
    dataloaders = create_dataloader(
        dataset=dataset_builder,
        batch_size=args.batch_size,
        train=True,
    )
    return dataloaders


def build_model(args):
    if args.arch == "gcn":
        print0("Using GCN model for LRGB task")
        model_config = GCNContactConfig(dense_repr=args.dense_repr)
        model = GCNContactModel(model_config).to("cuda")
        print0(model)
    else:
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
                enable_adj_emb=True,
                enable_diffusion=False,
                n_edge_feat=3,
                edge_feat_vocab_size=120,
                task_level="e",
                n_decode_layers=4,
                node_feat_vocab_size=120,
                n_node_feat=9,
                supernode=True,
                dropout=args.dropout,
                norm_fn="ln",
            )
    
        model = FloydNet(ModelConfig(**model_config)).to("cuda")
        model.init_weights()
        print0(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(f"Total trainable parameters: {trainable_params / 1e6:.2f}M")

    start_epoch = 0

    if is_ddp_initialized():
        torch.distributed.barrier()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

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
            
    if is_master_process() and args.arch == "floydnet":
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

    criterion = FocalLoss()
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
        amp=True,
        grad_clip_val=args.grad_clip,
        grad_accumulation=args.grad_accumulation,
        eval_interval=args.eval_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        max_train_epoch_len=args.max_train_epoch_len,
        max_val_epoch_len=args.max_val_epoch_len,
        max_test_epoch_len=args.max_test_epoch_len,
    )
    print0("Starting training...")
    trainer.fit(logger, dataloaders, start_epoch=start_epoch, metric=EdgeMRR())

    compute_cleanup()
    print0("Finished all processes.")


if __name__ == '__main__':
    args = parse_args()
    main(args)