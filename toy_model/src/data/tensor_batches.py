"""
Pre-tokenised, pre-padded batches held as whole tensors on the training device.

The per-item DataLoader path (CompDataset + collate_pad) re-tokenises every row with a regex and
builds tensors in Python for every batch; for this tiny model that CPU work dominates an epoch
(~1 s per 25k rows on a GH200, versus ~0.1 s of GPU time). Encoding once and slicing index
permutations removes it. Batches carry exactly the keys collate_pad produces (input_ids,
target_ids, loss_mask, pad_mask, and type when present), so step_loss / evaluate_split_by_type
are unchanged.
"""

from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

from .dataset import CompDataset


class TensorBatches:
    def __init__(self, dataset: CompDataset, device: torch.device, pad_id: int = 0):
        n = len(dataset)
        items = [dataset[i] for i in range(n)]
        L = max(ex["length"] for ex in items)
        input_ids = torch.full((n, L), pad_id, dtype=torch.long)
        target_ids = torch.full((n, L), -100, dtype=torch.long)
        loss_mask = torch.zeros((n, L), dtype=torch.bool)
        pad_mask = torch.ones((n, L), dtype=torch.bool)
        types: List[str] = []
        lengths = np.zeros(n, dtype=np.int64)
        for i, ex in enumerate(items):
            l = ex["length"]
            lengths[i] = l
            input_ids[i, :l] = ex["input_ids"]
            target_ids[i, :l] = ex["target_ids"]
            for p in ex["loss_positions"]:
                loss_mask[i, p] = True
            pad_mask[i, :l] = False
            if "type" in ex:
                types.append(ex["type"])
        self.n = n
        self.lengths = lengths  # kept on the CPU so batch trimming needs no device sync
        self.device = device
        self.input_ids = input_ids.to(device)
        self.target_ids = target_ids.to(device)
        self.loss_mask = loss_mask.to(device)
        self.pad_mask = pad_mask.to(device)
        self.types: Optional[np.ndarray] = np.array(types, dtype=object) if types else None
        self.vocab = dataset.vocab

    def __len__(self) -> int:
        return self.n

    def batches(self, batch_size: int, shuffle: bool, generator: Optional[torch.Generator] = None
                ) -> Iterator[Dict]:
        order = torch.randperm(self.n, generator=generator) if shuffle else torch.arange(self.n)
        order_np = order.numpy()
        order_dev = order.to(self.device)
        for start in range(0, self.n, batch_size):
            idx = order_dev[start:start + batch_size]
            # trim to the longest row in this batch, like collate_pad does
            L = int(self.lengths[order_np[start:start + batch_size]].max())
            batch = {
                "input_ids": self.input_ids[idx, :L],
                "target_ids": self.target_ids[idx, :L],
                "loss_mask": self.loss_mask[idx, :L],
                "pad_mask": self.pad_mask[idx, :L],
            }
            if self.types is not None:
                batch["type"] = self.types[order_np[start:start + batch_size]].tolist()
            yield batch
