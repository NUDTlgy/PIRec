from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SegConfig:
    lambda_sem: float = 1.0
    lambda_time: float = 1.0
    lambda_thr: float = 1.0
    l_min: int | None = 2
    l_max: int | None = None
    eps: float = 1e-8


def _resolve_len_bounds(l_min: int | None, l_max: int | None) -> tuple[int, int | None]:
    min_len = 1 if l_min is None else max(int(l_min), 1)
    max_len = None if l_max is None else max(int(l_max), 1)
    if max_len is not None and min_len > max_len:
        min_len = max_len
    return min_len, max_len


def _welford_std(n: int, m2: float, eps: float) -> float:
    return math.sqrt(m2 / max(n - 1, 1) + eps)


def _block_ranges_from_boundaries(boundaries: List[int]) -> List[Tuple[int, int]]:
    starts = [i for i, b in enumerate(boundaries) if b == 1]
    ranges: List[Tuple[int, int]] = []
    for k, st in enumerate(starts):
        ed = starts[k + 1] - 1 if k + 1 < len(starts) else len(boundaries) - 1
        ranges.append((st, ed))
    return ranges


def segment_single_user(
    items: List[int],
    times: List[int],
    item_emb: torch.Tensor,
    cfg: SegConfig,
) -> Tuple[List[int], List[float], List[Tuple[int, int]]]:
    n = len(items)
    min_len, max_len = _resolve_len_bounds(cfg.l_min, cfg.l_max)
    if n == 0:
        return [], [], []
    boundaries = [0] * n
    scores = [0.0] * n
    boundaries[0] = 1
    if n == 1:
        return boundaries, scores, [(0, 0)]

    stat_n = 1
    stat_mu = 0.0
    stat_m2 = 0.0
    cur_start = 0

    for t in range(1, n):
        gamma = stat_mu + cfg.lambda_thr * _welford_std(stat_n, stat_m2, cfg.eps)
        cur_ids = torch.tensor(items[cur_start:t], dtype=torch.long, device=item_emb.device)
        center = item_emb[cur_ids].mean(dim=0)
        item_vec = item_emb[int(items[t])]
        d_sem = 1.0 - F.cosine_similarity(item_vec.unsqueeze(0), center.unsqueeze(0), dim=-1).item()

        block_len = t - cur_start
        if block_len > 1:
            delta_t = math.log1p(max(int(times[t]) - int(times[t - 1]), 0))
            hist_delta = [
                math.log1p(max(int(times[j]) - int(times[j - 1]), 0))
                for j in range(cur_start + 1, t)
            ]
            bar_delta = float(sum(hist_delta) / max(len(hist_delta), 1))
            d_time = abs(delta_t - bar_delta) / (bar_delta + cfg.eps)
        else:
            d_time = 0.0

        score = cfg.lambda_sem * d_sem + cfg.lambda_time * d_time
        scores[t] = float(score)
        split_by_score = score > gamma and block_len >= min_len
        split_by_len = max_len is not None and block_len >= max_len
        if split_by_score or split_by_len:
            boundaries[t] = 1
            cur_start = t

        new_n = stat_n + 1
        delta = score - stat_mu
        new_mu = stat_mu + delta / new_n
        stat_m2 += delta * (score - new_mu)
        stat_mu = new_mu
        stat_n = new_n

    return boundaries, scores, _block_ranges_from_boundaries(boundaries)


def segment_all_users(
    user_full_items: Dict[int, List[int]],
    user_full_times: Dict[int, List[int]],
    item_emb: torch.Tensor,
    cfg: SegConfig,
) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for uid, items in user_full_items.items():
        b, s, blocks = segment_single_user(items, user_full_times[uid], item_emb, cfg)
        out[int(uid)] = {"boundaries": b, "scores": s, "blocks": blocks}
    return out


def block_sequence_for_prefix(boundaries: List[int], prefix_end_idx: int) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    if prefix_end_idx < 0:
        return [], (0, -1)
    starts = [i for i, b in enumerate(boundaries[: prefix_end_idx + 1]) if b == 1]
    if not starts:
        starts = [0]
    complete = [(starts[i], starts[i + 1] - 1) for i in range(len(starts) - 1)]
    return complete, (starts[-1], prefix_end_idx)

