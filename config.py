from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch


@dataclass
class Config:
    project_dir: Path = Path(__file__).resolve().parent
    root_dir: Path = project_dir.parent
    dataset: str = ("book")

    yelp_dir: Path = Path("/data3/yangzhiwei/liuguiyang/Demo/HS-GAT-main/data/yelp")
    yelp_business_json: Path | None = None
    yelp_review_json: Path | None = None
    yelp_user_json: Path | None = None

    lastfm_dir: Path = Path("/data3/yangzhiwei/liuguiyang/Demo/Meto/fm")
    lastfm_user_artists_dat: Path | None = None
    lastfm_user_taggedartists_ts_dat: Path | None = None
    lastfm_tags_dat: Path | None = None

    book_dir: Path = Path("/data3/yangzhiwei/liuguiyang/Demo/Meto/ambook")
    book_meta_jsonl: Path | None = None
    book_review_jsonl: Path | None = None

    output_dir: Path = project_dir / "artifacts"
    model_dir: Path = project_dir / "checkpoints"

    max_reviews: int | None = None
    dataset_max_reviews: dict[str, int | None] = field(
        default_factory=lambda: {
            "yelp": 702648,
            "lastfm": None,
            "book":830472,
        }
    )
    min_user_inter: int = 5
    min_item_inter: int = 5
    dedup_user_item_pairs: bool = False

    hidden_dim: int = 64
    graph_layers: int = 2
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn_dim: int = 256
    dropout: float = 0.1
    time_bias_init: float = 0.1
    max_seq_len: int | None = None
    max_phase_len: int = 20
    num_state_prototypes: int = 16

    lambda_sem: float = 1.0
    lambda_time: float = 1.0
    lambda_thr: float = 1.0
    l_min: int | None = 2
    l_max: int | None = None
    eps: float = 1e-8
    boundary_temperature: float = 0.67
    boundary_hard: bool = False
    boundary_prior: float = 0.2
    boundary_loss_weight: float = 0.05
    state_consistency_weight: float = 0.05

    warmup_epochs: int = 50
    warmup_lr: float = 1e-3
    warmup_neg_samples: int = 1
    warmup_batch_size: int = 4096
    warmup_aug_drop: float = 0.2
    warmup_contrastive_temp: float = 0.2

    seed: int = 42
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 5
    batch_size: int = 256
    batch_log_every: int = 0
    neg_samples: int = 1
    eval_neg_samples: int = 100
    grad_clip: float = 5.0
    eval_every: int = 1
    device: str = "cuda"
    ks: List[int] = field(default_factory=lambda: [10, 20])
    train_num_workers: int = 0
    use_amp: bool = True
    amp_dtype: str = "fp16"
    enable_tf32: bool = True

    def __post_init__(self) -> None:
        self.dataset = self.dataset.lower().strip()
        if self.dataset in {"fm", "lastfm", "last-fm"}:
            self.dataset = "lastfm"
        if self.dataset not in {"yelp", "lastfm", "book"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        self._apply_dataset_defaults()
        self._fill_dataset_paths()
        self.sync_paths()

    def _apply_dataset_defaults(self) -> None:
        # LastFM is more numerically fragile in the current training setup,
        # so we use a more conservative default recipe for stability.
        if self.dataset == "lastfm":
            self.lr = 3e-4
            self.warmup_lr = 5e-4
            self.warmup_epochs = 20
            self.batch_size = 128
            self.batch_log_every = 0
            self.grad_clip = 2.0
            self.use_amp = False
        elif self.dataset == "book":
            self.lr = 2e-4
            self.warmup_lr = 5e-4
            self.warmup_epochs = 20
            self.batch_size = 128
            self.batch_log_every = 0
            self.grad_clip = 1.0
            self.boundary_loss_weight = 0.02
            self.state_consistency_weight = 0.02
            self.use_amp = False

    def _fill_dataset_paths(self) -> None:
        self.yelp_business_json = self.yelp_business_json or self.yelp_dir / "yelp_academic_dataset_business.json"
        self.yelp_review_json = self.yelp_review_json or self.yelp_dir / "yelp_academic_dataset_review.json"
        self.yelp_user_json = self.yelp_user_json or self.yelp_dir / "yelp_academic_dataset_user.json"
        self.lastfm_user_artists_dat = self.lastfm_user_artists_dat or self.lastfm_dir / "user_artists.dat"
        self.lastfm_user_taggedartists_ts_dat = (
            self.lastfm_user_taggedartists_ts_dat
            or self.lastfm_dir / "user_taggedartists-timestamps.dat"
        )
        self.lastfm_tags_dat = self.lastfm_tags_dat or self.lastfm_dir / "tags.dat"
        self.book_meta_jsonl = self.book_meta_jsonl or self.book_dir / "meta_Books.jsonl"
        self.book_review_jsonl = self.book_review_jsonl or self.book_dir / "Books.jsonl"

    def ensure_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.sync_paths()

    def sync_paths(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.model_dir = Path(self.model_dir)
        self.raw_interaction_csv = self.output_dir / f"raw_interactions_{self.data_cache_stem}.csv"
        self.kcore_interaction_csv = self.output_dir / f"kcore_interactions_{self.data_cache_stem}.csv"
        self.cache_path = self.output_dir / f"prepared_{self.data_cache_stem}.pkl"
        self.segments_path = self.output_dir / self.tagged_filename(f"segments_{self.segment_cache_tag}", ".pkl")
        self.train_log_path = self.output_dir / self.tagged_filename("train_log_bce", ".json")
        self.efficiency_log_path = self.output_dir / self.tagged_filename("efficiency_epochs_bce", ".csv")

    @staticmethod
    def _hash_tag(payload: dict, prefix: str, length: int = 10) -> str:
        txt = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        digest = hashlib.md5(txt.encode("utf-8")).hexdigest()[:length]
        return f"{prefix}{digest}"

    @property
    def effective_max_reviews(self) -> int | None:
        if self.max_reviews is not None:
            return self.max_reviews
        return self.dataset_max_reviews.get(self.dataset)

    @property
    def data_size_tag(self) -> str:
        limit = self.effective_max_reviews
        return "maxall" if limit is None else f"max{int(limit)}"

    @property
    def data_cache_tag(self) -> str:
        payload = {
            "dataset": self.dataset,
            "max_reviews": self.effective_max_reviews,
            "min_user_inter": self.min_user_inter,
            "min_item_inter": self.min_item_inter,
            "dedup_user_item_pairs": self.dedup_user_item_pairs,
            "yelp_review_json": str(self.yelp_review_json),
            "lastfm_user_artists_dat": str(self.lastfm_user_artists_dat),
            "book_review_jsonl": str(self.book_review_jsonl),
        }
        return self._hash_tag(payload, "data")

    @property
    def data_cache_stem(self) -> str:
        return f"{self.dataset}_{self.data_size_tag}_{self.data_cache_tag}"

    @property
    def segment_cache_tag(self) -> str:
        payload = {
            "data_cache_tag": self.data_cache_tag,
            "hidden_dim": self.hidden_dim,
            "graph_layers": self.graph_layers,
            "warmup_epochs": self.warmup_epochs,
            "warmup_lr": self.warmup_lr,
            "num_state_prototypes": self.num_state_prototypes,
            "boundary_temperature": self.boundary_temperature,
            "boundary_hard": self.boundary_hard,
            "boundary_prior": self.boundary_prior,
            "boundary_loss_weight": self.boundary_loss_weight,
            "state_consistency_weight": self.state_consistency_weight,
            "warmup_aug_drop": self.warmup_aug_drop,
            "warmup_contrastive_temp": self.warmup_contrastive_temp,
        }
        return self._hash_tag(payload, "seg")

    def tagged_filename(self, stem: str, ext: str) -> str:
        return f"{stem}_{self.dataset}{ext}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-scale Transformer sequential recommendation.")
    parser.add_argument("--dataset", choices=["yelp", "lastfm", "book"], default=None)
    parser.add_argument("--yelp-dir", type=Path, default=None)
    parser.add_argument("--lastfm-dir", type=Path, default=None)
    parser.add_argument("--book-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--graph-layers", type=int, default=None)
    parser.add_argument("--transformer-layers", type=int, default=None)
    parser.add_argument("--transformer-heads", type=int, default=None)
    parser.add_argument("--time-bias-init", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-phase-len", type=int, default=None)
    parser.add_argument("--num-state-prototypes", type=int, default=None)
    parser.add_argument("--neg-samples", type=int, default=None)
    parser.add_argument("--eval-neg-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--boundary-temperature", type=float, default=None)
    parser.add_argument("--boundary-hard", type=int, choices=[0, 1], default=None)
    parser.add_argument("--boundary-prior", type=float, default=None)
    parser.add_argument("--boundary-loss-weight", type=float, default=None)
    parser.add_argument("--state-consistency-weight", type=float, default=None)
    parser.add_argument("--warmup-aug-drop", type=float, default=None)
    parser.add_argument("--warmup-contrastive-temp", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default=None)
    parser.add_argument("--force-rebuild-data", action="store_true")
    parser.add_argument("--force-resegment", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config() if args.dataset is None else Config(dataset=args.dataset)
    if args.yelp_dir is not None:
        cfg.yelp_dir = args.yelp_dir
        cfg.yelp_business_json = None
        cfg.yelp_review_json = None
        cfg.yelp_user_json = None
    if args.lastfm_dir is not None:
        cfg.lastfm_dir = args.lastfm_dir
        cfg.lastfm_user_artists_dat = None
        cfg.lastfm_user_taggedartists_ts_dat = None
        cfg.lastfm_tags_dat = None
    if args.book_dir is not None:
        cfg.book_dir = args.book_dir
        cfg.book_meta_jsonl = None
        cfg.book_review_jsonl = None
    cfg._fill_dataset_paths()
    for name in [
        "output_dir",
        "model_dir",
        "max_reviews",
        "epochs",
        "warmup_epochs",
        "batch_size",
        "batch_log_every",
        "eval_every",
        "hidden_dim",
        "graph_layers",
        "transformer_layers",
        "transformer_heads",
        "time_bias_init",
        "max_seq_len",
        "max_phase_len",
        "num_state_prototypes",
        "neg_samples",
        "eval_neg_samples",
        "lr",
        "warmup_lr",
        "dropout",
        "device",
        "seed",
        "amp_dtype",
        "boundary_temperature",
        "boundary_prior",
        "boundary_loss_weight",
        "state_consistency_weight",
        "warmup_aug_drop",
        "warmup_contrastive_temp",
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.boundary_hard is not None:
        cfg.boundary_hard = bool(int(args.boundary_hard))
    cfg.sync_paths()
    return cfg


CFG = Config()
