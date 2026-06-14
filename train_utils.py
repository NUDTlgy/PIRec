from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from data import EvalSample, TrainSample


class TrainDataset(Dataset):
    def __init__(
        self,
        samples: List[TrainSample],
        user_train_items: Dict[int, List[int]],
        user_train_times: Dict[int, List[int]],
        max_seq_len: int | None = None,
    ) -> None:
        self.samples = samples
        self.user_train_items = {int(k): np.asarray(v, dtype=np.int64) for k, v in user_train_items.items()}
        self.user_train_times = {int(k): np.asarray(v, dtype=np.int64) for k, v in user_train_times.items()}
        self.max_seq_len = None if max_seq_len is None else max(1, int(max_seq_len))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        uid = int(sample.uid)
        prefix_len = int(sample.prefix_len)
        start = 0 if self.max_seq_len is None else max(0, prefix_len - self.max_seq_len)
        return {
            "uid": uid,
            "target": int(sample.target),
            "prefix_end_idx": prefix_len - 1,
            "prefix_items": self.user_train_items[uid][start:prefix_len],
            "prefix_times": self.user_train_times[uid][start:prefix_len],
        }


def collate_train_batch(batch: List[dict], fixed_pad_len: int | None = None) -> dict:
    max_len = max((len(x["prefix_items"]) for x in batch), default=1)
    if fixed_pad_len is not None:
        max_len = max(1, int(fixed_pad_len))
    prefix_items = torch.zeros((len(batch), max_len), dtype=torch.long)
    prefix_times = torch.zeros((len(batch), max_len), dtype=torch.long)
    lengths = torch.zeros((len(batch),), dtype=torch.long)
    for i, sample in enumerate(batch):
        items = sample["prefix_items"][-max_len:]
        times = sample["prefix_times"][-max_len:]
        L = len(items)
        lengths[i] = L
        if L > 0:
            prefix_items[i, :L] = torch.as_tensor(items, dtype=torch.long)
            prefix_times[i, :L] = torch.as_tensor(times, dtype=torch.long)
    return {
        "uid": torch.tensor([x["uid"] for x in batch], dtype=torch.long),
        "target": torch.tensor([x["target"] for x in batch], dtype=torch.long),
        "prefix_end_idx": torch.tensor([x["prefix_end_idx"] for x in batch], dtype=torch.long),
        "prefix_items_padded": prefix_items,
        "prefix_times_padded": prefix_times,
        "prefix_lengths": lengths.clamp_min(1),
    }


def collate_eval_batch(batch: List[EvalSample], fixed_pad_len: int | None = None) -> dict:
    proxy = [
        {
            "uid": x.uid,
            "target": x.target,
            "prefix_items": x.prefix_items,
            "prefix_times": x.prefix_times,
            "prefix_end_idx": x.prefix_end_idx,
        }
        for x in batch
    ]
    return collate_train_batch(proxy, fixed_pad_len=fixed_pad_len)


def sample_negative(uid: int, num_items: int, seen: set) -> int:
    del uid
    if len(seen) >= num_items:
        return random.randint(0, max(num_items - 1, 0))
    while True:
        item = random.randint(0, num_items - 1)
        if item not in seen:
            return item


def sample_negatives_batch(
    user_ids: torch.Tensor,
    num_items: int,
    seen_history: Dict[int, set],
    n_neg: int,
    device: torch.device,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    if n_neg <= 0:
        return torch.empty((user_ids.numel(), 0), dtype=torch.long, device=device)
    batch_size = int(user_ids.numel())
    neg_np = np.random.randint(0, num_items, size=(batch_size, n_neg), dtype=np.int64)
    if num_items <= 1:
        return torch.zeros((batch_size, n_neg), dtype=torch.long, device=device)
    user_ids_cpu = user_ids.detach().cpu().tolist()
    targets_cpu = None if targets is None else targets.detach().cpu().tolist()
    for row, uid in enumerate(user_ids_cpu):
        seen = seen_history.get(int(uid), set())
        if targets_cpu is not None:
            seen = set(seen)
            seen.add(int(targets_cpu[row]))
        for col in range(n_neg):
            if int(neg_np[row, col]) in seen:
                neg_np[row, col] = sample_negative(int(uid), num_items, seen)
    return torch.as_tensor(neg_np, dtype=torch.long, device=device)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
