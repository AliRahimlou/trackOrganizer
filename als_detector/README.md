# ALS-Trained 1.1.1 Detector

This package learns Ableton `1.1.1` anchors from manually warped `.als` files, then predicts and writes anchors for new tracks.

## Scripts

- `extract_dataset.py`: parse `.als` and build JSONL labels `(audio_path -> target_sec)`
- `features.py`: extract cached features (mel, RMS, onset, low-band energy)
- `train.py`: train compact CNN anchor classifier
- `infer.py`: predict anchor for new audio and write warped `.als`
- `write_als.py`: write a known predicted anchor into template `.als`

## Install (project venv)

```bash
source venv/bin/activate
pip install -r als_detector/requirements-ml.txt
```

`madmom` is optional.

## 1) Extract labels from manual ALS

```bash
python als_detector/extract_dataset.py \
  --als "/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26 copy 2.als" \
  --out als_detector/data/labels.jsonl
```

Default behavior collapses repeated clips to one canonical target per audio file (earliest musically valid cluster).

## 2) Extract features

```bash
python als_detector/features.py \
  --dataset als_detector/data/labels.jsonl \
  --cache-dir als_detector/data/features \
  --manifest als_detector/data/features_manifest.jsonl
```

## 3) Train model

```bash
python als_detector/train.py \
  --dataset als_detector/data/labels.jsonl \
  --manifest als_detector/data/features_manifest.jsonl \
  --out-model als_detector/models/anchor_cnn.pt \
  --out-metrics als_detector/outputs/train_metrics.json
```

## 4) Infer on new track and write ALS

```bash
python als_detector/infer.py \
  --audio "/path/to/track.flac" \
  --template "/path/to/template.als" \
  --model als_detector/models/anchor_cnn.pt \
  --out als_detector/outputs/track_warped.als
```

Optional:

- `--use-madmom` for downbeat candidates/snapping
- `--plot als_detector/outputs/track_debug.png` for waveform + RMS marker view
- `--review-threshold 0.60` to flag low-confidence outputs

