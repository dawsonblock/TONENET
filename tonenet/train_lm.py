"""
Distributed training for Token Language Model.

Supports multi-GPU training with DistributedDataParallel.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from typing import Optional, List
import argparse

from .token_lm import TokenLanguageModel


class TokenDataset(Dataset):
    """Dataset of codec token sequences."""
    
    def __init__(self, path: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.data = torch.load(path)
        
        # Flatten all sequences into one
        if isinstance(self.data, list):
            self.data = torch.cat([t.flatten() for t in self.data])
        else:
            self.data = self.data.flatten()
    
    def __len__(self):
        return max(1, len(self.data) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        seq = self.data[start:end]
        
        # Pad if needed
        if len(seq) < self.seq_len:
            seq = torch.cat([seq, torch.zeros(self.seq_len - len(seq)).long()])
        
        return seq.long()


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train(
    data_path: str,
    output_path: str = "token_lm.pt",
    batch_size: int = 32,
    lr: float = 3e-4,
    steps: int = 100000,
    warmup_steps: int = 4000,
    seq_len: int = 512,
    save_every: int = 1000,
    distributed: bool = False
):
    """
    Train token language model.
    
    Args:
        data_path: Path to token data (.pt file)
        output_path: Output checkpoint path
        batch_size: Batch size per GPU
        lr: Learning rate
        steps: Total training steps
        warmup_steps: LR warmup steps
        seq_len: Sequence length
        save_every: Save checkpoint every N steps
        distributed: Use distributed training
    """
    rank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if distributed:
        rank = setup_distributed()
        device = f"cuda:{rank}"
    
    # Model
    model = TokenLanguageModel().to(device)
    
    if distributed:
        model = DDP(model, device_ids=[rank])
    
    # Data
    dataset = TokenDataset(data_path, seq_len=seq_len)
    
    if distributed:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    # Training loop
    step = 0
    model.train()
    
    while step < steps:
        if distributed:
            sampler.set_epoch(step // len(loader))
        
        for batch in loader:
            if step >= steps:
                break
            
            batch = batch.to(device)
            
            inner_model = model.module if distributed else model
            loss = inner_model.loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if rank == 0 and step % 100 == 0:
                print(f"Step {step}/{steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if rank == 0 and step % save_every == 0 and step > 0:
                state = model.module.state_dict() if distributed else model.state_dict()
                torch.save(state, output_path)
                print(f"Saved checkpoint: {output_path}")
            
            step += 1
    
    # Final save
    if rank == 0:
        state = model.module.state_dict() if distributed else model.state_dict()
        torch.save(state, output_path)
        print(f"Training complete. Saved: {output_path}")
    
    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Token LM")
    parser.add_argument("--data", required=True, help="Token data path")
    parser.add_argument("--output", default="token_lm.pt")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    
    # Check if running with torchrun
    distributed = "LOCAL_RANK" in os.environ
    
    train(
        data_path=args.data,
        output_path=args.output,
        batch_size=args.batch,
        lr=args.lr,
        steps=args.steps,
        seq_len=args.seq_len,
        distributed=distributed
    )
