# alsdrop

ALS-supervised Ableton 1.1.1 drop-anchor learner.

This package learns from manually warped `.als` files (where `BeatTime=0` is correctly placed) and predicts/writes 1.1.1 for new tracks.

## Install

Use your existing project venv:

```bash
source venv/bin/activate
pip install -r alsdrop/requirements-ml.txt
```

`madmom` is optional. If missing, the system falls back to beat-based candidate generation.

## CLI

Run with module entry:

```bash
python -m alsdrop.cli --help
```

Subcommands:

```bash
python -m alsdrop.cli extract --als_dir /path/to/als --out alsdrop/data/dataset.jsonl
python -m alsdrop.cli tier --dataset alsdrop/data/dataset.jsonl --out-gold alsdrop/data/dataset_gold.jsonl --out-silver alsdrop/data/dataset_silver.jsonl
python -m alsdrop.cli features --dataset alsdrop/data/dataset.jsonl --cache_dir alsdrop/data/features --manifest alsdrop/data/features_manifest.jsonl
python -m alsdrop.cli train --dataset alsdrop/data/dataset.jsonl --features alsdrop/data/features_manifest.jsonl --out alsdrop/models/model.pt
python -m alsdrop.cli predict --audio /path/to/track.flac --model alsdrop/models/model.pt --out alsdrop/outputs/predicted.json
python -m alsdrop.cli write-als --template /path/to/template.als --audio /path/to/track.flac --pred alsdrop/outputs/predicted.json --out alsdrop/outputs/track_warped.als
python -m alsdrop.cli batch --audio_dir /path/to/audio --template /path/to/template.als --model alsdrop/models/model.pt --out_dir alsdrop/outputs/batch
```

## Training pipeline

1) Extract ALS labels

```bash
python -m alsdrop.cli extract \
  --als "/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26 copy 2.als" \
  --out alsdrop/data/dataset.jsonl
```

2) Build feature cache

```bash
python -m alsdrop.cli features \
  --dataset alsdrop/data/dataset.jsonl \
  --cache_dir alsdrop/data/features \
  --manifest alsdrop/data/features_manifest.jsonl
```

3) Build strict training tiers (recommended)

```bash
python -m alsdrop.cli tier \
  --dataset alsdrop/data/dataset.jsonl \
  --out-gold alsdrop/data/dataset_gold.jsonl \
  --out-silver alsdrop/data/dataset_silver.jsonl \
  --out-bronze alsdrop/data/dataset_bronze.jsonl \
  --report-html alsdrop/outputs/dataset_tier_report.html
```

4) Train (use GOLD by default)

```bash
python -m alsdrop.cli train \
  --dataset alsdrop/data/dataset_gold.jsonl \
  --features alsdrop/data/features_manifest.jsonl \
  --candidates_dir alsdrop/data/candidates \
  --out alsdrop/models/model.pt \
  --metrics alsdrop/outputs/train_metrics.json
```

5) Evaluate with failure diagnostics

```bash
python -m alsdrop.cli evaluate \
  --dataset alsdrop/data/dataset_gold.jsonl \
  --model alsdrop/models/model.pt \
  --out alsdrop/outputs/eval_metrics.json \
  --report-html alsdrop/outputs/eval_report.html \
  --debug-dir alsdrop/outputs/eval_debug
```

## Architecture summary

- Stage A: downbeat/bar candidates (madmom preferred, beat fallback)
- Stage B: candidate ranking model (CNN + Transformer)
- Stage C: offset refinement head (sub-beat alignment)
- Stage D: confidence calibration (temperature scaling)

The model is trained as a **ranking problem** (correct downbeat candidate outranks others) with an offset loss for local timing refinement.

## ALS safety

Generated ALS files are written from template structure and validated with:

```bash
python -m alsdrop.cli validate-als --als /path/to/output.als
```

Checks include:

- `BeatTime=0` marker presence
- expected target second match (when provided)
- duplicate `ListId` detection

## Tests

```bash
source venv/bin/activate
pytest -q alsdrop/tests
```
