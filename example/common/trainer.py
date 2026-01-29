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
import sys
import copy
import json
import time
from pathlib import Path
from typing import List, Tuple
from functools import partial
from contextlib import nullcontext

from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from common.utils import empty_cache, is_master_process, reduce_metrics, save_checkpoint, print0, is_ddp_initialized

def get_binary_f1(pred: torch.Tensor, true: torch.Tensor):
    assert pred.shape == true.shape and pred.ndim == 1
    # compute binary f1 without sklearn
    tp = ((pred == 1) & (true == 1)).sum().item()
    tn = ((pred == 0) & (true == 0)).sum().item()
    fp = ((pred == 1) & (true == 0)).sum().item()
    fn = ((pred == 0) & (true == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        max_epochs: int,
        amp: bool,
        critn,
        eval_interval: int,
        test_interval: int,
        save_interval: int,
        out_dir: str,
        max_train_epoch_len: int = 10000,
        max_val_epoch_len: int = 10000,
        max_test_epoch_len: int = 10000,
        diffusion_model = None,
        grad_accumulation: int = 1,
        grad_clip_val: float = 1.0,
        eval_fn: str = "default",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.amp = amp
        self.critn = critn
        self.max_train_epoch_len = max_train_epoch_len
        self.max_val_epoch_len = max_val_epoch_len
        self.max_test_epoch_len = max_test_epoch_len
        self.eval_interval = eval_interval
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.out_dir = Path(out_dir)
        self.diffusion_model = diffusion_model
        self.grad_accumulation = grad_accumulation
        self.grad_clip_val = grad_clip_val

        self.graph_preprocess = model.module.preprocess if hasattr(model, "module") else model.preprocess

        if eval_fn == "default":
            self.eval_fn = self._evaluate
        elif eval_fn == "TSP":
            self.eval_fn = self._evaluate_TSP
        else:
            raise ValueError(f"Unknown eval_fn {eval_fn}")

    def fit(
        self,
        logger,
        dataloader,
        start_epoch,
        metric=None,
    ):
        start_fit_time = time.time()
        self.critn = self.critn.cuda()
        min_val_loss = float('inf')
        min_test_loss = float('inf')

        for epoch in range(start_epoch, self.max_epochs):
            if isinstance(dataloader["train"].sampler, DistributedSampler):
                dataloader["train"].sampler.set_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            loss = self._train_epoch(dataloader["train"], epoch)
            logger.log({
                "train/loss": loss,
                "lr": current_lr,
                "epoch": epoch,
            })

            should_evaluate_this_epoch = (epoch == 0 or (epoch + 1) % self.eval_interval == 0 or epoch == self.max_epochs - 1)
            if should_evaluate_this_epoch:
                metrics = self.eval_fn(dataloader["val"], "Val", epoch, metric=metric)
                min_val_loss = min(min_val_loss, metrics["loss"])
                logger.log({f"val/{k}": v for k, v in metrics.items()})
                print("val metrics:", metrics)
                self.scheduler.step(loss)
                if (epoch + 1) % self.save_interval == 0 or epoch == self.max_epochs - 1:
                    save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, loss, self.out_dir / "checkpoints")

            should_test_this_epoch = ((epoch + 1) % self.test_interval == 0 or epoch == self.max_epochs - 1)
            if should_test_this_epoch:
                metrics = self.eval_fn(dataloader["test"], "Test", epoch, metric=metric)
                print("test metrics:", metrics)
                min_test_loss = min(min_test_loss, metrics["loss"])
                logger.log({f"test/{k}": v for k, v in metrics.items()})

        end_fit_time = time.time()
        print0(f"Total training time: {end_fit_time - start_fit_time:.2f} seconds")

    def test(
        self,
        dataloader,
        epoch,
        sample_count_per_case=1,
    ):
        loss = self.eval_fn(dataloader["test"], "Test", epoch, sample_count_per_case=sample_count_per_case, metric=metric)
        return loss

    def _train_epoch(self, loader, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0

        progress_bar = tqdm(loader, desc=f"Train Epoch {epoch+1}/{self.max_epochs}", leave=False, file=sys.stdout, total=min(len(loader), self.max_train_epoch_len), disable=not is_master_process())

        for batch_idx, graph_batch in enumerate(progress_bar):
            if batch_idx >= self.max_train_epoch_len:
                break
            graph_batch = graph_batch.cuda()
            graph_batch = self.graph_preprocess(graph_batch)
            if self.diffusion_model is not None:
                graph_batch = self.diffusion_model.q_sample(graph_batch)

            sync_context = self.model.no_sync() if (batch_idx + 1) % self.grad_accumulation != 0 else nullcontext()
            with sync_context:
                with torch.amp.autocast('cuda', enabled=self.amp, dtype=torch.bfloat16):
                    pred = self.model(graph_batch)
                loss, _ = self.critn(pred, graph_batch)
                if self.diffusion_model is not None:
                    loss = self.diffusion_model.noise_schedule_reweight(loss, graph_batch["t"])
                loss_sum += loss.item()
                loss_count += 1
                loss = loss / self.grad_accumulation
                loss.backward()

            if (batch_idx + 1) % self.grad_accumulation == 0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_val)
                self.optimizer.step()
                self.optimizer.zero_grad()

            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f'{loss_sum/loss_count:.5f}', lr=f"{current_lr:.3e}")

        progress_bar.close()
        return loss_sum / loss_count

    @torch.no_grad()
    def _evaluate(self, loader, stage_name, epoch, metric=None, **kwargs):
        self.model.eval()
        loss_sum = 0.0
        loss_count = 0
        loss_breakdown_sum = {}
        if metric is not None:
            metric.clean()

        max_len = self.max_val_epoch_len if stage_name.lower() == "val" else self.max_test_epoch_len
        progress_bar = tqdm(loader, desc=f"{stage_name} Epoch {epoch+1}/{self.max_epochs}", leave=False, file=sys.stdout, total=min(len(loader), max_len), disable=not is_master_process())

        for batch_idx, graph_batch in enumerate(progress_bar):
            if batch_idx >= max_len:
                break
            graph_batch = graph_batch.cuda()
            graph_batch = self.graph_preprocess(graph_batch)
            with torch.amp.autocast('cuda', enabled=self.amp, dtype=torch.bfloat16):
                pred = self.model(graph_batch)

            loss, loss_breakdown = self.critn(pred, graph_batch)
            for k, v in loss_breakdown.items():
                if k not in loss_breakdown_sum:
                    loss_breakdown_sum[k] = 0.0
                loss_breakdown_sum[k] += v
            loss_sum += loss.item()
            loss_count += 1

            if metric is not None:
                metric.add_batch(pred, graph_batch)

            progress_bar.set_postfix(loss=f'{loss.item():.5f}')

        progress_bar.close()

        metrics = {k: v / loss_count for k, v in loss_breakdown_sum.items()}
        metrics["loss"] = loss_sum / loss_count
        metrics = reduce_metrics(metrics)

        if metric is not None:
            new_metric = metric.compute()
            new_metric = reduce_metrics(new_metric, reduction="sum")
            for k, v in new_metric.items():
                if k != "sample_count" and "sample_count" in new_metric:
                    v = v / new_metric["sample_count"]
                metrics[k] = v

        return metrics

    
    @torch.no_grad()
    def _evaluate_TSP(self, loader, stage_name, epoch, sample_count_per_case=1, **kwargs):
        self.model.eval()
        out_dir = self.out_dir / f"{stage_name}_infer_results_epoch{epoch+1}"
        out_dir.mkdir(parents=True, exist_ok=True)

        rank = dist.get_rank() if is_ddp_initialized() else 0
        out_path = out_dir / f"rank{rank:02d}.json"
        results = []
        metrics = {}
        count = 0

        max_len = self.max_val_epoch_len if stage_name.lower() == "val" else self.max_test_epoch_len

        for cur_sample_idx in range(sample_count_per_case):
            sampler = loader.sampler
            if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                sampler.set_epoch(cur_sample_idx + epoch * 1000)

            progress_bar = tqdm(loader, desc=f"{stage_name} Epoch {epoch+1}/{self.max_epochs}, Sample {cur_sample_idx+1}/{sample_count_per_case}", leave=False, file=sys.stdout, total=min(len(loader), max_len), disable=not is_master_process())

            for batch_idx, graph_batch in enumerate(progress_bar):
                if batch_idx >= max_len:
                    break
                graph_batch = graph_batch.cuda()
                assert torch.unique(graph_batch.batch).shape[0] == 1, "only support batch size 1 for evaluation"
                n = graph_batch.x.shape[0]
                gt_len = graph_batch.gt_len.item()

                graph_batch = self.graph_preprocess(graph_batch)
                with torch.amp.autocast('cuda', enabled=self.amp, dtype=torch.bfloat16):
                    pred_adj = (self.diffusion_model.generate(graph_batch, self.model) > 0.5).long()
                pred_adj = pred_adj[0]  # remove batch dimension
                pred_adj = (pred_adj + pred_adj.T).clamp(max=1, min=0)  # make it symmetric

                pred_len = (graph_batch.adj_attr[0, :, :, 0] * pred_adj).sum().item() // 2
                node_degrees = pred_adj.sum(dim=1)
                connect_matrix = pred_adj.float()
                for _ in range(20):
                    connect_matrix = connect_matrix + torch.matmul(connect_matrix, connect_matrix)
                    connect_matrix = torch.clamp(connect_matrix, 0, 1)
                connect_ok = torch.all(connect_matrix > 0.5).item()
                degree_ok = torch.all(node_degrees == 2).item()
                is_valid = connect_ok and degree_ok
                is_best = is_valid and pred_len <= gt_len

                pred_adj = pred_adj.float()
                true_adj = graph_batch.adj_label[0]  # remove batch dimension        
                f1_score = get_binary_f1(pred_adj.flatten(), true_adj.flatten())

                count += 1
                results.append(dict(
                    gid=graph_batch.gid[0],
                    nodes=n,
                    gt_len=gt_len,
                    pred_len=pred_len,
                    is_valid=is_valid,
                    is_best=is_best,
                    f1_score=f1_score,
                ))

                cur_metric = dict(
                    degree_ok=1 if degree_ok else 0,
                    connect_ok=1 if connect_ok else 0,
                    is_valid=1 if is_valid else 0,
                    is_best=1 if is_best else 0,
                    pred_len=pred_len,
                    gt_len=gt_len,
                    f1_score=f1_score,
                )
                for k, v in cur_metric.items():
                    if k not in metrics:
                        metrics[k] = 0.0
                    metrics[k] += v

            progress_bar.close()

            # save intermediate results
            with open(out_path, "w") as f:
                json.dump(results, f, indent=4)


        metrics = {k: v / count for k, v in metrics.items()}
        metrics = reduce_metrics(metrics)
        # a synthetic loss for lr scheduler 
        metrics["loss"] = (1 - metrics["is_valid"]) + (1 - metrics["f1_score"]) + (1 - metrics["is_best"])

        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

        return metrics