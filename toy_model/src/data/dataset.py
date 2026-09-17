"""
PyTorch Dataset classes for knowledge graph training.

Supervision rule: the loss is taken on every *entity* token that directly follows a
*relation* token. For the functor task (<e><r1><r2><t>, <e><f><t>) this is exactly
the last token, as before; for the skills task it also covers width-k rows
(<e><r><t><e><r><t>...), which have k supervised positions.
"""

import json
import re
from typing import Dict, List

import torch
from torch.utils.data import Dataset


TOKEN_PATTERN = re.compile(r"<e_\d+>|<r_\d+>|<f>|<f_inv>")


def tokenize_strict(s: str) -> List[str]:
    """Tokenize string into entity/relation tokens."""
    toks = TOKEN_PATTERN.findall(s)
    if "".join(toks) != s:
        bad = s.replace("".join(toks), "")
        raise ValueError(f"Non-token residue: '{bad}' in '{s}'")
    return toks


def is_entity_token(tok: str) -> bool:
    return tok.startswith("<e_")


def answer_positions(tgt_tokens: List[str]) -> List[int]:
    """
    Indices j into the shifted target sequence (target_ids[j] = tgt_tokens[j + 1]) whose
    target token is an entity that directly follows a relation token.
    """
    return [
        j for j in range(len(tgt_tokens) - 1)
        if is_entity_token(tgt_tokens[j + 1]) and not is_entity_token(tgt_tokens[j])
    ]


class CompDataset(Dataset):
    """Dataset for compositional knowledge graph training."""
    
    def __init__(
        self,
        path_json: str,
        vocab_path: str,
        max_len: int,
        expect_type: bool = False
    ):
        """
        Args:
            path_json: Path to the JSON data file
            vocab_path: Path to the vocabulary JSON file
            max_len: Maximum sequence length
            expect_type: Whether to expect type labels in the data
        """
        with open(path_json, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2tok = {i: t for i, t in enumerate(self.vocab)}
        self.max_len = max_len
        self.expect_type = expect_type
    
    def __len__(self) -> int:
        return len(self.items)
    
    def encode(self, toks: List[str]) -> List[int]:
        """Convert tokens to indices."""
        return [self.tok2id[t] for t in toks]
    
    def decode(self, ids: List[int]) -> List[str]:
        """Convert indices to tokens."""
        return [self.id2tok[i] for i in ids]
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        tgt = tokenize_strict(item["target_text"])
        
        input_ids = self.encode(tgt[:-1])
        target_ids = self.encode(tgt[1:])
        positions = answer_positions(tgt) or [len(target_ids) - 1]
        
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "loss_positions": positions,
            "last_pos": positions[-1],
            "length": len(input_ids),
        }
        
        if self.expect_type:
            out["type"] = item.get("type", "unknown")
        
        return out


def collate_pad(batch: List[Dict], pad_id: int = 0) -> Dict:
    """
    Collate function with padding.
    
    Args:
        batch: List of samples from CompDataset
        pad_id: ID to use for padding
        
    Returns:
        Batched and padded tensors; loss_mask is True at every supervised position
        (see answer_positions), in row-major order.
    """
    B = len(batch)
    maxL = max(ex["length"] for ex in batch)
    
    input_ids = torch.full((B, maxL), pad_id, dtype=torch.long)
    target_ids = torch.full((B, maxL), -100, dtype=torch.long)
    loss_mask = torch.zeros((B, maxL), dtype=torch.bool)
    pad_mask = torch.ones((B, maxL), dtype=torch.bool)
    types = []
    
    for i, ex in enumerate(batch):
        L = ex["length"]
        input_ids[i, :L] = ex["input_ids"]
        target_ids[i, :L] = ex["target_ids"]
        for p in ex["loss_positions"]:
            loss_mask[i, p] = True
        pad_mask[i, :L] = False
        if "type" in ex:
            types.append(ex["type"])
    
    out = {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "pad_mask": pad_mask
    }
    
    if types:
        out["type"] = types
    
    return out
