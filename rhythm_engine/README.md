# Rhythm Engine

Goal: build a DJ-usable, millisecond-accurate, any-genre beat/downbeat engine.

## Build Plan

1. Provider ensemble
   - Run multiple beat/downbeat sources instead of trusting one model.
   - Current adapters: `librosa`, `track_organizer`, `stem_ensemble`,
     optional `madmom`, optional `beat_this`.
   - `stem_ensemble` detects sibling Demucs stems and fuses role-specific grids.

2. Multi-hypothesis fusion
   - Cluster beats/downbeats across providers with ms-level radii.
   - Apply learned provider weights to both event timing and fused BPM.
   - Keep provider support, spread, and confidence in JSON for audit/debug.
   - Preserve parallel half/double-time and downbeat-phase hypotheses instead of
     collapsing to one grid too early.
   - Score candidate grids against full-song onset, low-end, RMS, and flux
     evidence before selecting the final grid.
   - Fill missing beats and rebuild downbeat phase for stable-tempo grids while
     skipping repair on unstable/expressive material.
   - Bound repaired grids to the detected audio duration so exports do not
     include phantom beats after the file ends.

3. Sample-level micro-refinement
   - Refine coarse beat points to local attack boundaries.
   - Current implementation uses high-rate waveform envelope/onset backtracking.
   - When sibling stems exist, it prefers drums, then bass, then full/audio for
     cleaner attack timing.

4. Audition/debug artifacts
   - Render audio+click overlays for the final grid or top hypotheses.
   - Use this to catch confident-but-wrong half-time, double-time, and phase
     failures before they become launch markers.

5. Millisecond evaluation
   - Report median, mean, p90, p95, max absolute error in ms.
   - Report hit rates at 5ms, 10ms, 20ms, and 70ms.
   - Report continuity so a mostly-right grid with one bad section is visible.

6. Active learning
   - Use Ableton/Rekordbox/manual correction history as ground truth.
   - Train provider weights, hypothesis gates, and micro-refinement decisions from
     real DJ launch labels rather than public benchmark assumptions.
   - Fusion can already consume a provider-weight JSON file so learned weights
     can be applied without code changes.
   - When reference beats are provided, every provider and hypothesis is
     evaluated so `python -m rhythm_engine.learn_weights` can learn fusion
     weights from the resulting JSON files.

## CLI

```bash
python -m rhythm_engine /path/to/audio.wav --provider librosa --json rhythm.json
```

Optional reference beat files are newline-separated seconds:

```bash
python -m rhythm_engine track.wav \
  --reference-beats beats.txt \
  --reference-downbeats downbeats.txt
```

Render an audition click overlay:

```bash
python -m rhythm_engine track.wav --click-wav track_click.wav
```

Export final beatgrid rows:

```bash
python -m rhythm_engine track.wav --beatgrid-csv beats.csv --beatgrid-json beats.json
```

Benchmark a reference manifest:

```bash
python -m rhythm_engine.benchmark manifest.csv --output rhythm_benchmark.jsonl
python -m rhythm_engine.learn_weights rhythm_benchmark.jsonl --output provider_weights.json
```
