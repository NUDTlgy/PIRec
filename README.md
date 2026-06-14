# DualScaleTransformer

This directory contains a clean implementation of the current method:

- Graph contrastive pretraining on the global user-item interaction graph.
- Learnable interest-state discovery.
- Dual-scale modeling with a fine-grained Transformer and coarse state prototype memory.
- BCE training objective.
- Sampled evaluation with `eval_neg_samples=100` by default.

The global graph is a simple user-item bipartite interaction graph. It does not use item attributes, user attributes, or local phase graphs.

## Run

From `D:\Ddesk\TOTO投稿`:

```powershell
python DualScaleTransformer\train_bce.py --dataset yelp
python DualScaleTransformer\train_bce.py --dataset lastfm
python DualScaleTransformer\train_bce.py --dataset book
```

Useful overrides:

```powershell
python DualScaleTransformer\train_bce.py --dataset yelp --epochs 80 --warmup-epochs 50 --eval-neg-samples 100
python DualScaleTransformer\train_bce.py --dataset book --batch-size 512 --amp-dtype bf16
```

Recent efficiency-oriented changes in this implementation:

- Training samples now reuse shared user sequences instead of caching a full prefix copy for every target.
- Negative sampling and sampled evaluation scoring are batched to reduce Python overhead and peak memory pressure.
- Transformer causal masks are cached per device/sequence length to avoid repeated allocation.

The current version does not rely on cached heuristic segmentation. Interest states are discovered inside the model with a learnable boundary predictor and prototype memory.

Default raw data paths follow the three reference projects:

- Yelp: `/data3/yangzhiwei/liuguiyang/Demo/HS-GAT-main/data/yelp`
- LastFM: `/data3/yangzhiwei/liuguiyang/Demo/Meto/fm`
- Book: `/data3/yangzhiwei/liuguiyang/Demo/Meto/ambook`

You can override them with `--yelp-dir`, `--lastfm-dir`, or `--book-dir`.
