from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data import load_raw_interactions


DATASETS = ("yelp", "lastfm", "book")
MAX_FILTER_ROUNDS = 100


def _fmt_count(x: int | float) -> str:
    return f"{int(x):,}"


def _stats(df: pd.DataFrame) -> dict:
    users = int(df["user_id"].nunique()) if not df.empty else 0
    items = int(df["item_id"].nunique()) if not df.empty else 0
    interactions = int(len(df))
    return {
        "users": users,
        "items": items,
        "interactions": interactions,
        "avg_user_history_len": interactions / max(users, 1),
    }


def _record(rows: list[dict], cfg: Config, round_idx: int, stage: str, df: pd.DataFrame) -> None:
    row = {
        "dataset": cfg.dataset,
        "data_size_tag": cfg.data_size_tag,
        "max_reviews": cfg.effective_max_reviews,
        "min_user_inter": cfg.min_user_inter,
        "min_item_inter": cfg.min_item_inter,
        "dedup_user_item_pairs": int(bool(cfg.dedup_user_item_pairs)),
        "round": round_idx,
        "stage": stage,
        **_stats(df),
    }
    rows.append(row)
    print(
        f"[{cfg.dataset}][round {round_idx:02d}][{stage}] "
        f"users={_fmt_count(row['users'])} | "
        f"items={_fmt_count(row['items'])} | "
        f"interactions={_fmt_count(row['interactions'])} | "
        f"avg_user_history_len={row['avg_user_history_len']:.4f}",
        flush=True,
    )


def _prepare_raw_for_filter(cfg: Config, force_rebuild_raw: bool) -> pd.DataFrame:
    df = load_raw_interactions(cfg, force_rebuild=force_rebuild_raw)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    df["_order"] = np.arange(len(df), dtype=np.int64)
    return df


def _audit_dataset(dataset: str, force_rebuild_raw: bool) -> list[dict]:
    cfg = Config(dataset=dataset)
    cfg.ensure_dirs()
    rows: list[dict] = []
    print(
        f"\n==================== {cfg.dataset} | "
        f"min_user_inter={cfg.min_user_inter} | min_item_inter={cfg.min_item_inter} ====================",
        flush=True,
    )

    cur = _prepare_raw_for_filter(cfg, force_rebuild_raw=force_rebuild_raw)
    _record(rows, cfg, 0, "before_filter", cur)

    if cfg.dedup_user_item_pairs:
        cur = cur.sort_values(["user_id", "item_id", "timestamp", "_order"])
        cur = cur.drop_duplicates(subset=["user_id", "item_id"], keep="last").reset_index(drop=True)
        _record(rows, cfg, 0, "after_dedup_user_item", cur)

    for round_idx in range(1, MAX_FILTER_ROUNDS + 1):
        before_len = len(cur)
        user_keep = cur["user_id"].value_counts()
        cur = cur[cur["user_id"].isin(user_keep[user_keep >= cfg.min_user_inter].index)].copy()
        _record(rows, cfg, round_idx, "after_user_filter", cur)

        item_keep = cur["item_id"].value_counts()
        cur = cur[cur["item_id"].isin(item_keep[item_keep >= cfg.min_item_inter].index)].copy()
        _record(rows, cfg, round_idx, "after_item_filter", cur)

        if len(cur) == before_len:
            _record(rows, cfg, round_idx, "converged", cur)
            break
    else:
        _record(rows, cfg, MAX_FILTER_ROUNDS, "reached_max_rounds", cur)

    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "data_size_tag",
        "max_reviews",
        "min_user_inter",
        "min_item_inter",
        "dedup_user_item_pairs",
        "round",
        "stage",
        "users",
        "items",
        "interactions",
        "avg_user_history_len",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PIRec3sudu k-core filtering statistics.")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated datasets: yelp,lastfm,book.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force-rebuild-raw", action="store_true")
    args = parser.parse_args()

    datasets = [x.strip().lower() for x in str(args.datasets).split(",") if x.strip()]
    out_path = args.output
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Config().output_dir / f"filter_stats_{ts}.csv"

    all_rows: list[dict] = []
    for dataset in datasets:
        all_rows.extend(_audit_dataset(dataset, force_rebuild_raw=bool(args.force_rebuild_raw)))

    _write_csv(out_path, all_rows)
    print(f"\n[done] filter stats saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
