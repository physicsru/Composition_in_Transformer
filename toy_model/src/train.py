"""
Training script for Emergent Analogy experiment.
"""

import argparse
import collections
import json
import math
import os
import random
import time
from functools import partial

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data import CompDataset, collate_pad
from data.tensor_batches import TensorBatches
from data.streams import FixedStreams
from model import GPT2LikeEncoder


class WarmupThenConstant(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup followed by constant rate."""
    
    def __init__(self, opt, warmup_steps=2000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(opt, last_epoch)
    
    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = step / self.warmup_steps if step <= self.warmup_steps else 1.0
        return [base * scale for base in self.base_lrs]


def step_loss(model, batch, device, use_amp=True):
    """Compute loss and accuracy for a batch."""
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    pad_mask = batch.get("pad_mask", None)
    if pad_mask is not None:
        pad_mask = pad_mask.to(device)

    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(input_ids, pad_mask=pad_mask)
        B, L, V = logits.shape
        logits_f = logits.view(B * L, V)
        targets_f = target_ids.view(B * L)
        mask_f = loss_mask.view(B * L)
        sel_logits = logits_f[mask_f]
        sel_targets = targets_f[mask_f]
        loss = F.cross_entropy(sel_logits, sel_targets)

    with torch.no_grad():
        pred = sel_logits.argmax(dim=-1)
        acc = (pred == sel_targets).float().mean().item()
        # Calculate probability of correct token
        probs = F.softmax(sel_logits, dim=-1)
        prob = probs[torch.arange(len(sel_targets), device=sel_logits.device), sel_targets].mean().item()
    
    return loss, acc, prob


@torch.no_grad()
def evaluate_loader(model, loader, device, use_amp=True):
    """Evaluate model on a data loader."""
    if loader is None:
        return None
    model.eval()
    sum_loss, sum_acc, n = 0.0, 0.0, 0
    for batch in loader:
        loss, acc, _ = step_loss(model, batch, device, use_amp)
        bs = batch["input_ids"].size(0)
        sum_loss += loss.item() * bs
        sum_acc += acc * bs
        n += bs
    mean_ce = sum_loss / max(1, n)
    return {
        "CE": mean_ce,
        "PPL": math.exp(min(20.0, mean_ce)),
        "ACC": sum_acc / max(1, n),
        "N": n
    }


@torch.no_grad()
def evaluate_split_by_type(model, loader, device, use_amp=True):
    """Evaluate model and split results by data type."""
    model.eval()
    sums = {}
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        types = batch["type"]

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, pad_mask=pad_mask)
        
        B, L, V = logits.shape
        logits_f = logits.view(B * L, V)
        targets_f = target_ids.view(B * L)
        mask_f = loss_mask.view(B * L)
        sel_logits = logits_f[mask_f]
        sel_targets = targets_f[mask_f]
        
        per_ce = F.cross_entropy(sel_logits, sel_targets, reduction='none')
        per_acc = (sel_logits.argmax(dim=-1) == sel_targets).float()
        
        # Calculate probability of correct token
        probs = F.softmax(sel_logits, dim=-1)
        per_prob = probs[torch.arange(len(sel_targets), device=sel_logits.device), sel_targets]
        
        # A sequence may have several supervised positions (width-k rows); the masked
        # selections are in row-major order, so walk them sequence by sequence.
        n_pos = loss_mask.sum(dim=1).tolist()
        per_ce_l, per_acc_l, per_prob_l = per_ce.tolist(), per_acc.tolist(), per_prob.tolist()
        idx = 0
        for t, n in zip(types, n_pos):
            d = sums.setdefault(t, {"sum_ce": 0, "sum_acc": 0, "sum_prob": 0, "n": 0,
                                    "sum_seq_acc": 0, "n_seq": 0})
            seq_ok = 1.0
            for k in range(idx, idx + n):
                d["sum_ce"] += per_ce_l[k]
                d["sum_acc"] += per_acc_l[k]
                d["sum_prob"] += per_prob_l[k]
                d["n"] += 1
                if per_acc_l[k] < 1.0:
                    seq_ok = 0.0
            d["sum_seq_acc"] += seq_ok
            d["n_seq"] += 1
            idx += n
    
    metrics = {}
    for t, d in sums.items():
        mce = d["sum_ce"] / max(1, d["n"])
        metrics[t] = {
            "CE": mce,
            "PPL": math.exp(min(20.0, mce)),
            "ACC": d["sum_acc"] / max(1, d["n"]),            # per supervised position
            "SEQ_ACC": d["sum_seq_acc"] / max(1, d["n_seq"]),  # all positions of a row correct
            "PROB": d["sum_prob"] / max(1, d["n"]),
            "N": d["n"],
            "N_SEQ": d["n_seq"],
        }
    return metrics


def flatten_metrics(prefix, mdict):
    """Flatten metrics dictionary for logging."""
    flat = {}
    for t, m in mdict.items():
        for k, v in m.items():
            flat[f"{prefix}_{k}/{t}"] = v
    return flat


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def train_stream(config, model, optimizer, scheduler, train_ds, test_ds, make_test_dl, device):
    """Update-based training with fixed logical streams (config['stream']); see data/streams.py.

    Loss per update = sum of answer CEs over all drawn rows / config['stream_denominator'] (fixed, default 256).
    Evaluates every `eval_every_updates`, saves epoch{k:03d}.pt every `save_every_updates` (k = update // save_every),
    writes metrics.jsonl rows with epoch = eval index and global_step = update, plus manifest.json and train_log.jsonl.
    """
    data_dir = config["data_dir"]
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    train_rows = json.load(open(os.path.join(data_dir, "train.json")))
    tok2id = {t: i for i, t in enumerate(train_ds.vocab)}
    streams = FixedStreams(train_rows, meta, config["stream"], int(config.get("stream_seed", config["seed"])), tok2id, device)
    max_updates = int(config["max_updates"])
    eval_every = int(config.get("eval_every_updates", 5000))
    save_every = int(config.get("save_every_updates", eval_every))
    D = float(config.get("stream_denominator", 256))
    rows_per_update = sum(n for _, n in streams.spec)
    manifest = dict(mode="stream", stream=streams.manifest(), rows_per_update=rows_per_update, denominator=D,
                    max_updates=max_updates, eval_every_updates=eval_every, save_every_updates=save_every,
                    seed=config["seed"], stream_seed=int(config.get("stream_seed", config["seed"])),
                    init_from=config.get("init_from"), init_from_sha=_sha(config["init_from"]) if config.get("init_from") else None,
                    data_hashes={k: _sha(os.path.join(data_dir, f"{k}.json")) for k in ("train", "test", "meta", "vocab")},
                    optimizer=dict(name=str(config.get("optimizer", "adam")), lr=config["lr"], weight_decay=config["weight_decay"],
                                   warmup_steps=config["warmup_steps"]),
                    model={k: config[k] for k in ("d_model", "n_layer", "n_head", "dropout", "max_len")})
    with open(os.path.join(config["save_dir"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[stream] {streams.spec} -> {rows_per_update} rows/update, denominator {D}, unique rows {streams.counts}")
    metrics_path = os.path.join(config["save_dir"], "metrics.jsonl")
    open(metrics_path, "w").close()
    log_path = os.path.join(config["save_dir"], "train_log.jsonl")
    open(log_path, "w").close()
    best_val = float("inf")
    run_ce = collections.defaultdict(float); run_n = collections.defaultdict(int); run_correct = collections.defaultdict(int)
    t0 = time.time()
    ckpt_cfg = {k: config[k] for k in ("d_model", "n_layer", "n_head", "dropout", "max_len")}
    for u in range(1, max_updates + 1):
        model.train()
        batch, slices = streams.next_batch()
        logits = model(batch["input_ids"], pad_mask=batch["pad_mask"])
        B, L, V = logits.shape
        mask = batch["loss_mask"]
        sel_logits = logits[mask]                       # rows are ordered by stream, one answer each
        sel_targets = batch["target_ids"][mask]
        ce = F.cross_entropy(sel_logits, sel_targets, reduction="none")
        loss = ce.sum() / D
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            correct = (sel_logits.argmax(-1) == sel_targets)
            off = 0
            for cat, n in slices:
                run_ce[cat] += float(ce[off:off + n].sum()); run_correct[cat] += int(correct[off:off + n].sum()); run_n[cat] += n
                off += n
        if u % 100 == 0 or u == 1:
            rec = dict(update=u, loss=float(loss), lr=optimizer.param_groups[0]["lr"], elapsed=time.time() - t0,
                       **{f"ce/{c}": run_ce[c] / max(1, run_n[c]) for c in run_n},
                       **{f"acc/{c}": run_correct[c] / max(1, run_n[c]) for c in run_n})
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            run_ce.clear(); run_n.clear(); run_correct.clear()
        if u % eval_every == 0 or u == max_updates:
            metrics = evaluate_split_by_type(model, make_test_dl(), device, config["use_amp"])
            ce_macro = sum(m["CE"] for m in metrics.values()) / max(1, len(metrics))
            acc_macro = sum(m["ACC"] for m in metrics.values()) / max(1, len(metrics))
            k = u // eval_every
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": k, "global_step": u, "train_ce": float(loss), "macro_ce": ce_macro,
                                    "macro_acc": acc_macro, "val": metrics}) + "\n")
            print(f"update {u} | loss {float(loss):.4f} | macro ACC {acc_macro:.3f} | {time.time() - t0:.0f}s")
            ckpt = {"model": model.state_dict(), "config": ckpt_cfg, "vocab": train_ds.vocab, "epoch": k, "update": u}
            if u % save_every == 0:
                torch.save(ckpt, os.path.join(config["save_dir"], f"epoch{u // save_every:03d}.pt"))
            if ce_macro < best_val:
                best_val = ce_macro
                torch.save(ckpt, os.path.join(config["save_dir"], "best.pt"))
    manifest["exposures"] = streams.exposure_report()
    manifest["updates_done"] = max_updates
    with open(os.path.join(config["save_dir"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("Training completed (stream mode)!")
    return model


def train(config):
    """Main training function."""
    # PyYAML reads scientific notation without a dot (e.g. `lr: 1e-4`) as a string.
    for key in ("lr", "weight_decay"):
        if key in config and config[key] is not None:
            config[key] = float(config[key])
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # Data loading
    data_dir = config["data_dir"]
    vocab_path = os.path.join(data_dir, "vocab.json")
    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")
    
    train_ds = CompDataset(train_path, vocab_path, max_len=config["max_len"], expect_type=False)
    test_ds = CompDataset(test_path, vocab_path, max_len=config["max_len"], expect_type=True)
    
    eval_bs = int(config.get("eval_batch_size") or config["batch_size"])
    if config.get("data_loader", "tensor") == "tensor":
        # Encode once, keep everything on the device, slice index permutations (see data/tensor_batches.py).
        train_tb = TensorBatches(train_ds, device)
        test_tb = TensorBatches(test_ds, device)
        shuffle_gen = torch.Generator().manual_seed(config["seed"])
        make_train_dl = lambda: train_tb.batches(config["batch_size"], shuffle=True, generator=shuffle_gen)
        make_test_dl = lambda: test_tb.batches(eval_bs, shuffle=False)
    else:
        collate_fn = partial(collate_pad, pad_id=0)
        train_dl = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True,
            num_workers=config.get("num_workers", 4), collate_fn=collate_fn, pin_memory=True)
        test_dl = DataLoader(
            test_ds, batch_size=eval_bs, shuffle=False,
            num_workers=config.get("num_workers", 4), collate_fn=collate_fn, pin_memory=True)
        make_train_dl = lambda: train_dl
        make_test_dl = lambda: test_dl
    
    # Model setup
    model = GPT2LikeEncoder(
        len(train_ds.vocab),
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        dropout=config["dropout"],
        max_len=config["max_len"]
    ).to(device)
    
    if config.get("init_from"):
        ck = torch.load(config["init_from"], map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        print(f"Initialised weights from {config['init_from']} (epoch {ck.get('epoch')}); optimizer state is fresh")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    opt_name = str(config.get("optimizer", "adam")).lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}' (use 'adam' or 'adamw')")
    scheduler = WarmupThenConstant(optimizer, warmup_steps=config["warmup_steps"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    if config.get("stream"):
        with open(os.path.join(config["save_dir"], "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        return train_stream(config, model, optimizer, scheduler, train_ds, test_ds, make_test_dl, device)
    
    # Optional W&B logging
    wandb = None
    if config.get("use_wandb", False):
        try:
            import wandb as _wandb
            wandb = _wandb
            mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
            
            # Get project/run name from env vars (priority: env > config > default)
            wandb_project = os.environ.get("WANDB_PROJECT") or config.get("project", "emergent_analogy")
            wandb_run_name = os.environ.get("WANDB_RUN_NAME") or config.get("run_name") or f"run_{int(time.time())}"
            
            print(f"[W&B] project: {wandb_project}, run: {wandb_run_name}")
            
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                mode=mode,
                config=config,
            )
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("lr", step_metric="global_step")
        except Exception as e:
            print(f"[W&B] disabled: {e}")
            wandb = None
    
    # Training loop
    metrics_path = os.path.join(config["save_dir"], "metrics.jsonl")
    open(metrics_path, "w").close()
    with open(os.path.join(config["save_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    best_val = float("inf")
    global_step = 0
    log_every = config.get("log_every", 50)
    
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running, n_ex = 0, 0
        
        for i, batch in enumerate(make_train_dl(), 1):
            loss, acc, prob = step_loss(model, batch, device, config["use_amp"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            bs = batch["input_ids"].size(0)
            running += loss.item() * bs
            n_ex += bs
            global_step += 1
            
            if wandb and (i % log_every == 0 or i == 1):
                wandb.log({
                    "global_step": global_step,
                    "train/CE_last": loss.item(),
                    "train/ACC_last": acc,
                    "train/PROB_last": prob,
                    "lr": optimizer.param_groups[0]["lr"]
                }, step=global_step)
        
        train_ce = running / max(1, n_ex)
        
        if epoch % config.get("eval_every", 10) == 0:
            print(f"epoch {epoch} | train CE: {train_ce:.4f}")
        
        # Evaluation
        metrics = evaluate_split_by_type(model, make_test_dl(), device, config["use_amp"])
        ce_macro = sum(m["CE"] for m in metrics.values()) / max(1, len(metrics))
        acc_macro = sum(m["ACC"] for m in metrics.values()) / max(1, len(metrics))
        
        if epoch % config.get("eval_every", 10) == 0:
            print(f"\n== epoch {epoch} validation ==")
            for k, v in sorted(metrics.items()):
                print(f"{k:>25}: CE={v['CE']:.4f} | PPL={v['PPL']:.3f} | ACC={v['ACC']:.3f} | SEQ={v['SEQ_ACC']:.3f} | PROB={v['PROB']:.4f} | N={v['N']}")
            print(f"{'macro(CE)':>25}: CE={ce_macro:.4f}")
            print(f"{'macro(ACC)':>25}: ACC={acc_macro:.3f}\n")
        
        if wandb:
            log_payload = {
                "global_step": global_step,
                "epoch": epoch,
                "train/CE_epoch": train_ce,
                "val/macro/CE": ce_macro,
                "val/macro/ACC": acc_macro
            }
            log_payload.update(flatten_metrics("val", metrics))
            wandb.log(log_payload, step=global_step)
        
        # Always keep a local per-epoch record (one JSON object per line) next to the checkpoints.
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "global_step": global_step, "train_ce": train_ce,
                "macro_ce": ce_macro, "macro_acc": acc_macro, "val": metrics,
            }) + "\n")
        
        # Save checkpoint (skip if save_every <= 0)
        save_every = config.get("save_every", 0)
        if save_every > 0:
            ckpt = {
                "model": model.state_dict(),
                "config": {
                    "d_model": config["d_model"],
                    "n_layer": config["n_layer"],
                    "n_head": config["n_head"],
                    "dropout": config["dropout"],
                    "max_len": config["max_len"],
                },
                "vocab": train_ds.vocab,
                "epoch": epoch,
            }
            
            if epoch % save_every == 0 or epoch in set(config.get("save_epochs") or []):
                torch.save(ckpt, os.path.join(config["save_dir"], f"epoch{epoch:03d}.pt"))
            
            if ce_macro < best_val:
                best_val = ce_macro
                torch.save(ckpt, os.path.join(config["save_dir"], "best.pt"))
    
    if wandb:
        wandb.finish()
    
    print("Training completed!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Emergent Analogy model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override save directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Override checkpoint interval in epochs (0 = no checkpoints)")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Batch size for evaluation (default: batch_size); evaluation is forward-only")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Override weight decay")
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"],
                        help="Override optimizer (adam = coupled L2, adamw = decoupled)")
    parser.add_argument("--stream", type=str, default=None,
                        help="fixed logical streams per update, e.g. tt:128,atomic:64,first:32,second:32 (update-based training; see data/streams.py)")
    parser.add_argument("--max_updates", type=int, default=None)
    parser.add_argument("--eval_every_updates", type=int, default=None)
    parser.add_argument("--save_every_updates", type=int, default=None)
    parser.add_argument("--stream_denominator", type=float, default=None, help="fixed loss denominator (default 256)")
    parser.add_argument("--stream_seed", type=int, default=None)
    parser.add_argument("--save_epochs", type=str, default=None,
                        help="comma-separated extra epochs at which to save a checkpoint (in addition to save_every; needs save_every > 0)")
    parser.add_argument("--init_from", type=str, default=None,
                        help="checkpoint (.pt) whose model weights initialise training; optimizer/scheduler start fresh")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override training seed (model init and shuffling)")
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name")
    parser.add_argument("--wandb_run", type=str, default=None,
                        help="WandB run name")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.save_dir:
        config["save_dir"] = args.save_dir
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.eval_batch_size:
        config["eval_batch_size"] = args.eval_batch_size
    if args.save_every is not None:
        config["save_every"] = args.save_every
    if args.lr:
        config["lr"] = args.lr
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.optimizer:
        config["optimizer"] = args.optimizer
    if args.seed is not None:
        config["seed"] = args.seed
    if args.init_from:
        config["init_from"] = args.init_from
    if args.save_epochs:
        config["save_epochs"] = [int(x) for x in args.save_epochs.split(",") if x.strip()]
    for key in ("stream", "max_updates", "eval_every_updates", "save_every_updates", "stream_denominator", "stream_seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("d_model", "n_layer", "n_head"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.no_wandb:
        config["use_wandb"] = False
    if args.wandb_project:
        config["project"] = args.wandb_project
    if args.wandb_run:
        config["run_name"] = args.wandb_run
    
    train(config)


if __name__ == "__main__":
    main()
