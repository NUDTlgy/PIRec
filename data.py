from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config


def _fmt_count(x: int) -> str:
    return f"{int(x):,}"


@dataclass
class TrainSample:
    uid: int
    target: int
    prefix_len: int

@dataclass
class EvalSample:
    uid: int
    target: int
    prefix_items: List[int]
    prefix_times: List[int]
    prefix_end_idx: int


@dataclass
class PreparedData:
    num_users: int
    num_items: int
    user_full_items: Dict[int, List[int]]
    user_full_times: Dict[int, List[int]]
    user_train_items: Dict[int, List[int]]
    user_train_times: Dict[int, List[int]]
    user_train_history: Dict[int, set]
    user_valid_history: Dict[int, set]
    train_samples: List[TrainSample]
    valid_eval_samples: List[EvalSample]
    test_eval_samples: List[EvalSample]
    train_edges: List[Tuple[int, int]]


def _print_prepared_summary(prepared: PreparedData, tag: str = "[data] prepared") -> None:
    return


def _prepared_cache_is_compatible(prepared: PreparedData) -> bool:
    if not prepared.train_samples:
        return True
    sample = prepared.train_samples[0]
    return hasattr(sample, "prefix_len")


@lru_cache(maxsize=200000)
def _to_timestamp(date_str: str) -> int:
    if not date_str:
        return 0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(date_str[: len(fmt)], fmt).timestamp())
        except Exception:
            pass
    return 0


def _load_yelp_raw(cfg: Config) -> pd.DataFrame:
    if not cfg.yelp_business_json.exists():
        raise FileNotFoundError(f"Missing Yelp business json: {cfg.yelp_business_json}")
    if not cfg.yelp_review_json.exists():
        raise FileNotFoundError(f"Missing Yelp review json: {cfg.yelp_review_json}")

    valid_items: set[str] = set()
    with open(cfg.yelp_business_json, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            bid = obj.get("business_id")
            if bid:
                valid_items.add(str(bid))

    rows: List[tuple[str, str, int, float]] = []
    with open(cfg.yelp_review_json, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if cfg.effective_max_reviews is not None and idx > cfg.effective_max_reviews:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            uid = obj.get("user_id")
            bid = obj.get("business_id")
            if not uid or not bid or str(bid) not in valid_items:
                continue
            rows.append((str(uid), str(bid), _to_timestamp(obj.get("date", "")), float(obj.get("stars", 0.0))))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "rating"])


def _load_book_raw(cfg: Config) -> pd.DataFrame:
    if not cfg.book_meta_jsonl.exists():
        raise FileNotFoundError(f"Missing book meta jsonl: {cfg.book_meta_jsonl}")
    if not cfg.book_review_jsonl.exists():
        raise FileNotFoundError(f"Missing book review jsonl: {cfg.book_review_jsonl}")

    valid_items: set[str] = set()
    with open(cfg.book_meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            bid = obj.get("parent_asin") or obj.get("asin")
            if bid:
                valid_items.add(str(bid).strip())

    rows: List[tuple[str, str, int, float]] = []
    with open(cfg.book_review_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if cfg.effective_max_reviews is not None and idx > cfg.effective_max_reviews:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            uid = obj.get("user_id")
            bid = obj.get("parent_asin") or obj.get("asin")
            if not uid or not bid:
                continue
            bid = str(bid).strip()
            if bid not in valid_items:
                continue
            ts = obj.get("timestamp", obj.get("time", 0))
            try:
                ts_int = int(ts)
            except Exception:
                ts_int = 0
            rows.append((str(uid).strip(), bid, ts_int, float(obj.get("rating", obj.get("stars", 0.0)))))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "rating"])


def _load_lastfm_raw(cfg: Config) -> pd.DataFrame:
    if not cfg.lastfm_user_artists_dat.exists():
        raise FileNotFoundError(f"Missing LastFM user_artists.dat: {cfg.lastfm_user_artists_dat}")
    if not cfg.lastfm_user_taggedartists_ts_dat.exists():
        raise FileNotFoundError(f"Missing LastFM timestamp file: {cfg.lastfm_user_taggedartists_ts_dat}")

    ua = pd.read_csv(cfg.lastfm_user_artists_dat, sep="\t")
    ut = pd.read_csv(cfg.lastfm_user_taggedartists_ts_dat, sep="\t")
    ua = ua.rename(columns={"userID": "user_id", "artistID": "item_id", "weight": "rating"})
    ut = ut.rename(columns={"userID": "user_id", "artistID": "item_id"})
    ut["timestamp"] = pd.to_numeric(ut["timestamp"], errors="coerce").fillna(0).astype(np.int64) // 1000
    ts = ut.groupby(["user_id", "item_id"], as_index=False)["timestamp"].max()
    df = ua.merge(ts, on=["user_id", "item_id"], how="left")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    df = df[df["timestamp"] > 0].copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = pd.to_numeric(df.get("rating", 1.0), errors="coerce").fillna(1.0).astype(float)
    return df[["user_id", "item_id", "timestamp", "rating"]]


def build_raw_interaction_csv(cfg: Config, force_rebuild: bool = False) -> Path:
    cfg.ensure_dirs()
    out_path = cfg.raw_interaction_csv
    if out_path.exists() and not force_rebuild:
        return out_path

    if cfg.dataset == "yelp":
        df = _load_yelp_raw(cfg)
    elif cfg.dataset == "lastfm":
        df = _load_lastfm_raw(cfg)
    else:
        df = _load_book_raw(cfg)

    if df.empty:
        raise RuntimeError(f"No raw interactions loaded for dataset={cfg.dataset}.")
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return out_path


def load_raw_interactions(cfg: Config, force_rebuild: bool = False) -> pd.DataFrame:
    path = build_raw_interaction_csv(cfg, force_rebuild=force_rebuild)
    df = pd.read_csv(path)
    expected = {"user_id", "item_id", "timestamp", "rating"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"raw csv missing columns: {sorted(missing)}")
    return df


def load_kcore_interactions(cfg: Config, force_rebuild: bool = False) -> pd.DataFrame:
    path = cfg.kcore_interaction_csv
    if path.exists() and not force_rebuild:
        return pd.read_csv(path)

    df = load_raw_interactions(cfg, force_rebuild=force_rebuild)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    df["_order"] = np.arange(len(df), dtype=np.int64)
    if cfg.dedup_user_item_pairs:
        df = df.sort_values(["user_id", "item_id", "timestamp", "_order"])
        df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last").reset_index(drop=True)
    df = _iterative_kcore(df, cfg.min_user_inter, cfg.min_item_inter)
    if df.empty:
        raise RuntimeError("No interactions remain after k-core filtering.")
    df = df.sort_values(["user_id", "timestamp", "_order"]).reset_index(drop=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def _iterative_kcore(df: pd.DataFrame, min_user_inter: int, min_item_inter: int) -> pd.DataFrame:
    cur = df.copy()
    for _ in range(100):
        before = len(cur)
        user_keep = cur["user_id"].value_counts()
        cur = cur[cur["user_id"].isin(user_keep[user_keep >= min_user_inter].index)]
        item_keep = cur["item_id"].value_counts()
        cur = cur[cur["item_id"].isin(item_keep[item_keep >= min_item_inter].index)]
        if len(cur) == before:
            break
    return cur.reset_index(drop=True)


def _remap_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    user_vals = sorted(out["user_id"].unique().tolist())
    item_vals = sorted(out["item_id"].unique().tolist())
    user_map = {u: i for i, u in enumerate(user_vals)}
    item_map = {p: i for i, p in enumerate(item_vals)}
    out["user_id"] = out["user_id"].map(user_map).astype(int)
    out["item_id"] = out["item_id"].map(item_map).astype(int)
    return out


def _build_sequences(df: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    user_items: Dict[int, List[int]] = {}
    user_times: Dict[int, List[int]] = {}
    for uid, grp in df.groupby("user_id", sort=False):
        user_items[int(uid)] = grp["item_id"].astype(int).tolist()
        user_times[int(uid)] = grp["timestamp"].astype(int).tolist()
    return user_items, user_times


def _split_train_valid_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["user_id", "timestamp", "_order"]).reset_index(drop=True)
    test_idx = ordered.groupby("user_id", sort=False).tail(1).index
    remain = ordered.drop(index=test_idx).copy()
    valid_idx = remain.groupby("user_id", sort=False).tail(1).index
    return (
        remain.drop(index=valid_idx).copy().reset_index(drop=True),
        remain.loc[valid_idx].copy().reset_index(drop=True),
        ordered.loc[test_idx].copy().reset_index(drop=True),
    )


def _build_train_samples(
    user_train_items: Dict[int, List[int]],
    user_train_times: Dict[int, List[int]],
) -> List[TrainSample]:
    del user_train_times
    samples: List[TrainSample] = []
    for uid, items in user_train_items.items():
        for i in range(1, len(items)):
            samples.append(
                TrainSample(
                    uid=int(uid),
                    target=int(items[i]),
                    prefix_len=i,
                )
            )
    return samples


def _build_eval_samples(
    user_prefix_items: Dict[int, List[int]],
    user_prefix_times: Dict[int, List[int]],
    target_by_user: Dict[int, int],
) -> List[EvalSample]:
    samples: List[EvalSample] = []
    for uid, target in target_by_user.items():
        prefix_items = user_prefix_items.get(uid, [])
        prefix_times = user_prefix_times.get(uid, [])
        if not prefix_items:
            continue
        samples.append(
            EvalSample(
                uid=int(uid),
                target=int(target),
                prefix_items=[int(x) for x in prefix_items],
                prefix_times=[int(x) for x in prefix_times],
                prefix_end_idx=len(prefix_items) - 1,
            )
        )
    return samples


def prepare_data(cfg: Config, force_rebuild: bool = False) -> PreparedData:
    cfg.ensure_dirs()
    if cfg.cache_path.exists() and not force_rebuild:
        with open(cfg.cache_path, "rb") as f:
            prepared = pickle.load(f)
        if not _prepared_cache_is_compatible(prepared):
            return prepare_data(cfg, force_rebuild=True)
        _print_prepared_summary(prepared, tag="[data] cache summary")
        return prepared

    df = load_kcore_interactions(cfg, force_rebuild=force_rebuild)
    df = _remap_ids(df)
    df = df.sort_values(["user_id", "timestamp", "_order"]).reset_index(drop=True)

    train_df, valid_df, test_df = _split_train_valid_test(df)
    full_items, full_times = _build_sequences(df)
    train_items, train_times = _build_sequences(train_df)

    test_target = {int(r.user_id): int(r.item_id) for r in test_df.itertuples(index=False)}
    valid_target = {int(r.user_id): int(r.item_id) for r in valid_df.itertuples(index=False)}
    valid_ts = {int(r.user_id): int(r.timestamp) for r in valid_df.itertuples(index=False)}
    users = sorted(test_target.keys())
    full_items = {u: full_items[u] for u in users if u in full_items}
    full_times = {u: full_times[u] for u in users if u in full_times}
    train_items = {u: train_items.get(u, []) for u in users}
    train_times = {u: train_times.get(u, []) for u in users}

    valid_prefix_items = {u: train_items[u] + [valid_target[u]] for u in users if u in valid_target}
    valid_prefix_times = {u: train_times[u] + [valid_ts[u]] for u in users if u in valid_target}
    train_edges = [
        (int(uid), int(item))
        for uid, items in train_items.items()
        for item in items
    ]
    prepared = PreparedData(
        num_users=int(df["user_id"].max()) + 1,
        num_items=int(df["item_id"].max()) + 1,
        user_full_items=full_items,
        user_full_times=full_times,
        user_train_items=train_items,
        user_train_times=train_times,
        user_train_history={u: set(seq) for u, seq in train_items.items()},
        user_valid_history={u: set(valid_prefix_items.get(u, train_items.get(u, []))) for u in users},
        train_samples=_build_train_samples(train_items, train_times),
        valid_eval_samples=_build_eval_samples(train_items, train_times, valid_target),
        test_eval_samples=_build_eval_samples(valid_prefix_items, valid_prefix_times, test_target),
        train_edges=train_edges,
    )

    with open(cfg.cache_path, "wb") as f:
        pickle.dump(prepared, f)
    _print_prepared_summary(prepared)
    return prepared

