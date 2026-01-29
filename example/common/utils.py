import os
import gc
import random
import datetime
import functools
from pathlib import Path

import numpy as np

import torch
from torch import distributed as dist


def is_master_process() -> bool:
    ddp_rank = int(os.environ.get('RANK', 0))
    return ddp_rank == 0

def print0(s="",**kwargs):
    if is_master_process():
        print(s, **kwargs)

def rank_zero_only_(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if is_master_process():
            return fn(*args, **kwargs)
        return None
    return wrapped_fn

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

def compute_init(seed: int):
    """Basic initialization that we keep doing over and over, so make common."""

    device_type = "cuda"
    assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA"

    seed_everything(seed)

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested:
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device, timeout=datetime.timedelta(seconds=3600))
        dist.barrier()
    else:
        device = torch.device(device_type)

    print0(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()

def empty_cache():
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

def reduce_metrics(metric_dict, reduction="mean"):
    if not is_ddp_initialized():
        return metric_dict

    metric_names = sorted(metric_dict.keys())
    metric_values = torch.tensor([metric_dict[k] for k in metric_names], dtype=torch.float32, device="cuda")

    dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)

    if reduction == "mean":
        world_size = dist.get_world_size()
        metric_values /= world_size
    elif reduction == "sum":
        pass  # already summed
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")

    avg_metric_dict = {k: v.item() for k, v in zip(metric_names, metric_values)}
    return avg_metric_dict

@rank_zero_only_
def save_checkpoint(model, optimizer, scheduler, epoch, current_val_error, output_dir):
    checkpoint = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_error': current_val_error, 
    }

    checkpoint_filename = f'epoch_{epoch+1:05d}.pt' 
    checkpoint_path = Path(output_dir) / checkpoint_filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True) 

    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', -1) + 1 

    return model, optimizer, scheduler, start_epoch
