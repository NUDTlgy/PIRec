from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from config import Config, build_arg_parser, config_from_args, seed_everything
from data import EvalSample, PreparedData, prepare_data
from graph import InteractionGraph, build_interaction_graph
from model import DualScaleTransformerRec
from train_utils import collate_eval_batch, sample_negatives_batch


@torch.inference_mode()
def evaluate_sampled(
    model: DualScaleTransformerRec,
    graph: InteractionGraph,
    prepared: PreparedData,
    eval_samples: List[EvalSample],
    cfg: Config,
    device: torch.device,
    seen_history: Dict[int, set],
    batch_size: int = 128,
) -> Dict[str, float]:
    model.eval()
    user_states, item_states = model.encode_graph(graph)
    ks = list(cfg.ks)
    max_k = max(ks)
    totals = {k: {"recall": 0.0, "ndcg": 0.0} for k in ks}
    total_count = 0

    for st in range(0, len(eval_samples), batch_size):
        chunk = eval_samples[st : st + batch_size]
        batch = collate_eval_batch(chunk, fixed_pad_len=model.max_seq_len)
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        user_h = model.encode_users(batch, user_states, item_states)
        uids = torch.tensor([int(sample.uid) for sample in chunk], dtype=torch.long, device=device)
        targets = torch.tensor([int(sample.target) for sample in chunk], dtype=torch.long, device=device)
        negs = sample_negatives_batch(
            uids,
            prepared.num_items,
            seen_history,
            int(cfg.eval_neg_samples),
            device=device,
            targets=targets,
        )

        cands = torch.cat([targets.unsqueeze(1), negs], dim=1)
        scores = model.score_targets(user_h, cands, user_states, item_states, uids)
        k_use = min(max_k, int(cands.size(1)))
        top_idx = torch.topk(scores, k=k_use, dim=1).indices
        hit_pos = top_idx.eq(0).float()
        found = hit_pos.any(dim=1)
        rank = torch.full((cands.size(0),), k_use + 1, dtype=torch.long, device=device)
        if found.any():
            rank[found] = hit_pos[found].argmax(dim=1) + 1
        total_count += int(rank.numel())
        for k in ks:
            hit = rank <= k
            ndcg = torch.where(
                hit,
                1.0 / torch.log2(rank.to(torch.float32) + 1.0),
                torch.zeros_like(rank, dtype=torch.float32),
            )
            totals[k]["recall"] += float(hit.to(torch.float32).sum().item())
            totals[k]["ndcg"] += float(ndcg.sum().item())

    out = {}
    for k in ks:
        denom = max(total_count, 1)
        out[f"Recall@{k}"] = totals[k]["recall"] / denom
        out[f"NDCG@{k}"] = totals[k]["ndcg"] / denom
    return out


def _print_metrics(tag: str, metrics: dict, ks: list[int]) -> None:
    rec = "  ".join(f"Recall@{k}={metrics.get(f'Recall@{k}', 0.0):.6f}" for k in ks)
    ndcg = "  ".join(f"NDCG@{k}={metrics.get(f'NDCG@{k}', 0.0):.6f}" for k in ks)
    print(tag)
    print(rec)
    print(ndcg)


def main(cfg: Config, checkpoint: Path, force_rebuild_data: bool = False) -> None:
    from train import build_model

    cfg.ensure_dirs()
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    prepared = prepare_data(cfg, force_rebuild=force_rebuild_data)
    graph = build_interaction_graph(prepared.num_users, prepared.num_items, prepared.train_edges).to(device)
    model = build_model(cfg, prepared).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    metrics = evaluate_sampled(
        model,
        graph,
        prepared,
        prepared.test_eval_samples,
        cfg,
        device,
        seen_history=prepared.user_valid_history,
        batch_size=max(8, cfg.batch_size // 2),
    )
    _print_metrics(f"[test][checkpoint={checkpoint}][sampled neg={cfg.eval_neg_samples}]", metrics, list(cfg.ks))


def build_eval_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Evaluate a PIRec checkpoint on the test set."
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser


if __name__ == "__main__":
    parser = build_eval_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    main(cfg, args.checkpoint, force_rebuild_data=args.force_rebuild_data)
