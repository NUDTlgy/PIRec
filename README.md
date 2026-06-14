# PIRec

Code for **Graph-Pretrained Dual-Granularity Prototype Interest Learning for Sequential Recommendation**.

PIRec first learns collaborative user/item representations with graph contrastive pretraining on a global user-item interaction graph. It then combines a time-aware fine-grained sequence encoder with a boundary-aware coarse-grained interest state discovery module and a shared prototype memory for next-item recommendation.

## Files

```text
config.py        Arguments, paths, and hyperparameters
data.py          Raw data loading, k-core filtering, ID mapping, and train/valid/test split
graph.py         Global user-item graph construction and LightGCN-style propagation
pretraining.py   Graph contrastive pretraining
model.py         PIRec model: time-aware encoder, boundary learning, state discovery, prototypes, fusion, scoring
train_utils.py   Training/evaluation datasets, batching, and negative sampling
train.py         Training entry point; uses validation set for checkpoint selection
eval.py          Test entry point; loads a saved checkpoint and evaluates on the test set
```


## Requirements

The code is implemented with Python and PyTorch. A typical environment includes:

```bash
pip install torch numpy pandas
```

Optional:

```bash
pip install psutil
```

## Data

By default, the code expects datasets under:

```text
../data/yelp
../data/lastfm
../data/book
```

Expected raw files:

```text
Yelp:
  yelp_academic_dataset_business.json
  yelp_academic_dataset_review.json

LastFM:
  user_artists.dat
  user_taggedartists-timestamps.dat

Book:
  meta_Books.jsonl
  Books.jsonl
```

You can override paths from the command line:

```bash
python train.py --dataset yelp --yelp-dir /path/to/yelp
python train.py --dataset lastfm --lastfm-dir /path/to/lastfm
python train.py --dataset book --book-dir /path/to/book
```

Processed caches and checkpoints are written to:

```text
artifacts/
checkpoints/
```

## Training

Run from this directory:

```bash
python train.py --dataset yelp
python train.py --dataset lastfm
python train.py --dataset book
```

During training, only the validation set is evaluated for checkpoint selection. The terminal output is intentionally minimal and reports only training loss.

Useful options:

```bash
python train.py --dataset book --epochs 80 --batch-size 128
python train.py --dataset yelp --warmup-epochs 50 --eval-neg-samples 100
python train.py --dataset book --max-reviews 830472 --force-rebuild-data
python train.py --dataset yelp --device cuda --amp-dtype bf16
```

## Test Evaluation

After training, evaluate the test set by loading a checkpoint:

```bash
python eval.py --dataset book --checkpoint checkpoints/best_model_xxx_book.pt
```

Use the same dataset/path options as training when needed:

```bash
python eval.py --dataset yelp --yelp-dir /path/to/yelp --checkpoint checkpoints/best_model_xxx_yelp.pt
```

## Main Hyperparameters

Common options:

```text
--hidden-dim                 Embedding dimension
--graph-layers               Number of graph propagation layers
--transformer-layers         Number of time-aware Transformer layers
--transformer-heads          Number of attention heads
--num-state-prototypes       Number of shared interest prototypes
--boundary-prior             Target boundary activation rate
--boundary-temperature       Gumbel/slot temperature for differentiable boundary learning
--warmup-epochs              Graph contrastive pretraining epochs
--warmup-aug-drop            Edge dropout ratio for graph view augmentation
--warmup-contrastive-temp    Contrastive temperature
--eval-neg-samples           Number of sampled negatives for validation/test ranking
```

Dataset-specific defaults are defined in `config.py`.

## Notes

- Training uses the validation set to select the best checkpoint.
- Test metrics are reported only by `eval.py`.
- The global graph is a user-item bipartite graph built from training interactions.
- The coarse-grained branch uses learnable boundary discovery and prototype memory inside `model.py`.
- The implementation does not use item attributes, user attributes, or hand-crafted local phase graphs.
