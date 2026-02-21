"""
data/pipeline.py
Phase 2 — Data Pipeline
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class AyoDataset(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  {len(df):,} positions loaded.")

        #Fast Numpy Memory
        self.board = df[[f"hole_{i}" for i in range(12)]].values.astype(np.int64)

        self.context = np.stack([
            df["score_0"].values / 48.0,
            df["score_1"].values / 48.0,
            df["ply"].values     / 200.0,
        ], axis=1).astype(np.float32)

        self.turn = df["turn"].values.astype(np.int64)

        self.policy = df[[f"policy_{i}" for i in range(6)]].values.astype(np.float32)
        
        # Normalize policy to ensure strict probability distribution
        row_sums = self.policy.sum(axis=1, keepdims=True)
        self.policy = np.divide(self.policy, row_sums, out=self.policy, where=row_sums!=0)

        #"Greedy Ayo" Fourth-Root Scaling
        raw_value = df["value"].values.astype(np.float32)
        self.value = np.sign(raw_value) * (np.abs(raw_value) ** 0.25)

        self.capture = df["capture_next"].values.astype(np.float32)

        self.next_state = (
            df[[f"next_hole_{i}" for i in range(12)]].values.astype(np.float32)
            / 48.0
        )

        #Stats
        print(f"  Board range   : {self.board.min()} – {self.board.max()} seeds")
        print(f"  Value range   : {self.value.min():.3f} – {self.value.max():.3f}")
        print(f"  Capture rate  : {self.capture.mean():.1%}")
        print(f"  Turn 0 / 1    : {(self.turn==0).sum():,} / {(self.turn==1).sum():,}")
        print("  Ready.\n")

    def __len__(self) -> int:
        return len(self.board)

    def __getitem__(self, idx: int):
        inputs = {
            "board"  : torch.from_numpy(self.board[idx]),
            "turn"   : torch.tensor(self.turn[idx],   dtype=torch.long),
            "context": torch.from_numpy(self.context[idx]),
        }
        targets = {
            "policy"    : torch.from_numpy(self.policy[idx]),
            "value"     : torch.tensor(self.value[idx],   dtype=torch.float32),
            "capture"   : torch.tensor(self.capture[idx], dtype=torch.float32),
            "next_state": torch.from_numpy(self.next_state[idx]),
        }
        return inputs, targets


def make_loaders(csv_path    : str,
                 val_split   : float = 0.05,
                 batch_size  : int   = 256,
                 num_workers : int   = 4,
                 seed        : int   = 42,
                 ) -> tuple[DataLoader, DataLoader]:
    dataset = AyoDataset(csv_path)
    n       = len(dataset)
    n_val   = int(n * val_split)
    n_train = n - n_val

    rng = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=rng
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train : {n_train:,} samples  ({len(train_loader):,} batches)")
    print(f"Val   : {n_val:,} samples  ({len(val_loader):,} batches)")
    return train_loader, val_loader
