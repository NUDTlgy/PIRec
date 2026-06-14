from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, build_arg_parser, config_from_args, seed_everything
from data import PreparedData, prepare_data
from eval import evaluate_sampled
from graph import build_interaction_graph
from model import DualScaleTransformerRec
from train_utils import TrainDataset, collate_train_batch, sample_negatives_batch, seed_worker
from warmup import warm_up_interaction_graph


def _fmt_count(x: int) -> str:
    return f"{int(x):,}"


def _resolve_max_seq_len(cfg: Config, prepared: PreparedData) -> int:
    if cfg.max_seq_len is not None and int(cfg.max_seq_len) > 0:
        return int(cfg.max_seq_len)
    return max(1, max((len(x) for x in prepared.user_full_items.values()), default=1))


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_cuda_peak_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_memory_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "gpu_mem_allocated_mib": None,
            "gpu_mem_reserved_mib": None,
            "gpu_mem_peak_allocated_mib": None,
            "gpu_mem_peak_reserved_mib": None,
        }
    return {
        "gpu_mem_allocated_mib": torch.cuda.memory_allocated(device) / 1024**2,
        "gpu_mem_reserved_mib": torch.cuda.memory_reserved(device) / 1024**2,
        "gpu_mem_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "gpu_mem_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def _cpu_rss_mib() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        return None


def _append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def _jsonable_config(cfg: Config) -> dict:
    d = asdict(cfg)
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def _print_metrics(tag: str, metrics: dict, ks: list[int]) -> None:
    rec = "  ".join(f"Recall@{k}={metrics.get(f'Recall@{k}', 0.0):.6f}" for k in ks)
    ndcg = "  ".join(f"NDCG@{k}={metrics.get(f'NDCG@{k}', 0.0):.6f}" for k in ks)
    print(tag)
    print(rec)
    print(ndcg)


def _resolve_amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if str(name).lower().strip() == "bf16" else torch.float16


def _autocast_ctx(use_amp: bool, amp_dtype: torch.dtype):
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
    return torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)


def _build_grad_scaler(use_scaler: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    return torch.cuda.amp.GradScaler(enabled=use_scaler)


def build_model(cfg: Config, prepared: PreparedData) -> DualScaleTransformerRec:
    max_seq_len = _resolve_max_seq_len(cfg, prepared)
    return DualScaleTransformerRec(
        num_users=prepared.num_users,
        num_items=prepared.num_items,
        hidden_dim=cfg.hidden_dim,
        graph_layers=cfg.graph_layers,
        transformer_layers=cfg.transformer_layers,
        transformer_heads=cfg.transformer_heads,
        transformer_ffn_dim=cfg.transformer_ffn_dim,
        dropout=cfg.dropout,
        time_bias_init=cfg.time_bias_init,
        max_seq_len=max_seq_len,
        max_phase_len=cfg.max_phase_len,
        num_state_prototypes=cfg.num_state_prototypes,
        boundary_temperature=cfg.boundary_temperature,
        boundary_hard=cfg.boundary_hard,
        boundary_prior=cfg.boundary_prior,
    )


def main(cfg: Config, force_rebuild_data: bool = False, force_resegment: bool = False) -> None:
    cfg.ensure_dirs()
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.enable_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.enable_tf32)

    print("==================== DualScaleTransformer BCE ====================")
    print(
        f"[run] dataset={cfg.dataset} | device={device} | eval_neg_samples={cfg.eval_neg_samples} | "
        f"lr={cfg.lr} | batch_size={cfg.batch_size} | max_seq_len={cfg.max_seq_len} | use_amp={cfg.use_amp}"
    )
    run_t0 = time.time()
    data_t0 = time.time()
    prepared = prepare_data(cfg, force_rebuild=force_rebuild_data)
    data_prepare_sec = time.time() - data_t0
    graph = build_interaction_graph(prepared.num_users, prepared.num_items, prepared.train_edges).to(device)
    model = build_model(cfg, prepared).to(device)
    total_params, trainable_params = _count_parameters(model)
    print(
        "[model] built | "
        f"params={_fmt_count(total_params)} | trainable={_fmt_count(trainable_params)} | "
        f"max_seq_len={model.max_seq_len} | max_phase_len={model.max_phase_len} | "
        f"num_state_prototypes={cfg.num_state_prototypes}"
    )

    _sync_if_cuda(device)
    warmup_t0 = time.time()
    warm_up_interaction_graph(model, graph, prepared.train_edges, prepared.user_train_history, cfg, device)
    _sync_if_cuda(device)
    warmup_sec = time.time() - warmup_t0
    if force_resegment:
        print("[train] force_resegment is ignored in the learnable state discovery version.")

    dataset = TrainDataset(
        prepared.train_samples,
        prepared.user_train_items,
        prepared.user_train_times,
        max_seq_len=model.max_seq_len,
    )
    train_num_workers = max(0, int(cfg.train_num_workers))
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        collate_fn=lambda b: collate_train_batch(b, fixed_pad_len=model.max_seq_len),
        worker_init_fn=seed_worker,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(train_num_workers > 0),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = _build_grad_scaler(use_scaler)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    efficiency_log_path = Path(cfg.output_dir) / cfg.tagged_filename(f"efficiency_epochs_bce_{run_ts}", ".csv")
    efficiency_fields = [
        "run_ts",
        "dataset",
        "epoch",
        "loss",
        "train_sec",
        "eval_sec",
        "epoch_total_sec",
        "samples",
        "batches",
        "samples_per_s",
        "eval_users",
        "infer_users_per_s",
        "gpu_mem_allocated_mib",
        "gpu_mem_reserved_mib",
        "gpu_mem_peak_allocated_mib",
        "gpu_mem_peak_reserved_mib",
        "cpu_rss_mib",
        "total_params",
        "trainable_params",
        "batch_size",
        "max_seq_len",
        "eval_neg_samples",
        "use_amp",
        "amp_dtype",
        "boundary_rate",
        "valid/Recall@10",
        "valid/NDCG@10",
    ]
    train_log = {
        "config": _jsonable_config(cfg),
        "run": {
            "run_ts": run_ts,
            "dataset": cfg.dataset,
            "device": str(device),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "data_prepare_sec": data_prepare_sec,
            "warmup_sec": warmup_sec,
            "efficiency_log_path": str(efficiency_log_path),
        },
        "epochs": [],
    }
    best_val = -1.0
    best_path: Path | None = None
    n_neg = max(1, int(cfg.neg_samples))
    num_batches = len(loader)
    train_loop_t0 = time.time()

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        _sync_if_cuda(device)
        _reset_cuda_peak_stats(device)
        t0 = time.time()
        total_loss = 0.0
        n_batch = 0
        n_samples = 0
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(use_amp, amp_dtype):
                user_states, item_states = model.encode_graph(graph)
                user_h = model.encode_users(batch, user_states, item_states)
                pos_items = batch["target"]
                uids = batch["uid"]
                pos_scores = model.score_targets(user_h, pos_items, user_states, item_states, uids)

                neg_items = sample_negatives_batch(
                    uids,
                    prepared.num_items,
                    prepared.user_train_history,
                    n_neg,
                    device,
                    targets=pos_items,
                )
                neg_scores = model.score_candidate_set(user_h, neg_items, item_states).reshape(-1)

                rec_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                rec_loss = rec_loss + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                boundary_loss = model.latest_aux_losses.get("boundary_loss", rec_loss.new_zeros(()))
                state_consistency_loss = model.latest_aux_losses.get(
                    "state_consistency_loss",
                    rec_loss.new_zeros(()),
                )
                loss = rec_loss
                loss = loss + float(cfg.boundary_loss_weight) * boundary_loss
                loss = loss + float(cfg.state_consistency_weight) * state_consistency_loss
            if use_scaler:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            total_loss += float(loss.item())
            n_batch += 1
            n_samples += int(pos_items.size(0))
            should_log_batch = (
                int(cfg.batch_log_every) > 0
                and (batch_idx % int(cfg.batch_log_every) == 0 or batch_idx == num_batches)
            )
            if should_log_batch:
                elapsed = time.time() - t0
                print(
                    f"[train][epoch {epoch:03d}/{cfg.epochs:03d}] "
                    f"batch {batch_idx:04d}/{num_batches:04d} | "
                    f"loss={float(loss.item()):.4f} | "
                    f"samples={n_samples:,} | "
                    f"time={elapsed:.2f}s"
                )

        avg_loss = total_loss / max(n_batch, 1)
        _sync_if_cuda(device)
        epoch_time = time.time() - t0
        epoch_info = {
            "run_ts": run_ts,
            "dataset": cfg.dataset,
            "epoch": epoch,
            "loss": avg_loss,
            "train_sec": epoch_time,
            "time_s": epoch_time,
            "eval_sec": None,
            "epoch_total_sec": epoch_time,
            "samples": n_samples,
            "batches": n_batch,
            "samples_per_s": n_samples / max(epoch_time, 1e-8),
            "eval_users": None,
            "infer_users_per_s": None,
            "cpu_rss_mib": _cpu_rss_mib(),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "batch_size": cfg.batch_size,
            "max_seq_len": model.max_seq_len,
            "eval_neg_samples": cfg.eval_neg_samples,
            "use_amp": use_amp,
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "boundary_rate": float(model.latest_aux_losses.get("boundary_rate", torch.tensor(0.0)).item()),
        }
        epoch_info.update(_cuda_memory_stats(device))
        print(
            f"[train] epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"loss={avg_loss:.4f} | time={epoch_time:.2f}s | "
            f"samples/s={epoch_info['samples_per_s']:.2f} | "
            f"boundary_rate={epoch_info['boundary_rate']:.4f}"
        )

        if epoch % int(cfg.eval_every) == 0:
            _sync_if_cuda(device)
            eval_t0 = time.time()
            valid_metrics = evaluate_sampled(
                model,
                graph,
                prepared,
                prepared.valid_eval_samples,
                cfg,
                device,
                seen_history=prepared.user_train_history,
                batch_size=max(8, cfg.batch_size // 2),
            )
            _sync_if_cuda(device)
            eval_sec = time.time() - eval_t0
            eval_users = len(prepared.valid_eval_samples)
            epoch_info.update({f"valid/{k}": v for k, v in valid_metrics.items()})
            epoch_info["eval_sec"] = eval_sec
            epoch_info["epoch_total_sec"] = epoch_info["train_sec"] + eval_sec
            epoch_info["eval_users"] = eval_users
            epoch_info["infer_users_per_s"] = eval_users / max(eval_sec, 1e-8)
            epoch_info["cpu_rss_mib"] = _cpu_rss_mib()
            epoch_info.update(_cuda_memory_stats(device))
            _print_metrics(f"[valid][sampled neg={cfg.eval_neg_samples}]", valid_metrics, list(cfg.ks))
            print(
                f"[efficiency][eval] epoch={epoch} total_sec={eval_sec:.4f} "
                f"users={eval_users} users/s={epoch_info['infer_users_per_s']:.4f}",
                flush=True,
            )
            cur = valid_metrics.get("Recall@10", 0.0)
            if cur > best_val:
                best_val = cur
                best_path = Path(cfg.model_dir) / cfg.tagged_filename(f"best_model_{run_ts}_ep{epoch}", ".pt")
                torch.save(model.state_dict(), best_path)
                print(f"[checkpoint] best valid Recall@10={best_val:.6f} | {best_path}")

        _append_csv_row(efficiency_log_path, epoch_info, efficiency_fields)
        train_log["epochs"].append(epoch_info)

    last_path = Path(cfg.model_dir) / cfg.tagged_filename(f"last_model_{run_ts}", ".pt")
    torch.save(model.state_dict(), last_path)
    train_log["run"]["train_loop_wall_sec"] = time.time() - train_loop_t0
    train_log["run"]["total_wall_sec"] = time.time() - run_t0
    train_log["run"]["best_valid_recall_at_10"] = best_val
    train_log["run"]["best_model_path"] = str(best_path) if best_path is not None else None
    train_log["run"]["last_model_path"] = str(last_path)
    with open(cfg.train_log_path, "w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)
    print(f"[checkpoint] last model: {last_path}")
    print(f"[log] train log: {cfg.train_log_path}")
    print(f"[log] efficiency csv: {efficiency_log_path}")
    if best_path is not None:
        print(f"[done] best valid Recall@10={best_val:.6f} | {best_path}")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    main(cfg, force_rebuild_data=args.force_rebuild_data, force_resegment=args.force_resegment)
