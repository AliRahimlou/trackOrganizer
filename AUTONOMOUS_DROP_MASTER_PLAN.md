# Autonomous Drop Master Plan

Goal: increase autonomous drop-marker saves without lowering safety. The system should overgenerate candidates, hydrate them with stem and micro-boundary evidence, rank them, verify safe saves, and send only useful uncertainty cases to review.

## Current Operating Levels

- Level 1: AI places/refines a marker after a review action.
- Level 2: AI auto-saves only high-confidence tracks. Current system is here.
- Level 3: AI auto-saves high plus strong medium tracks, with low/disagreement cases reviewed.
- Level 4: AI processes most of the library; review is mostly audit and edge cases.
- Level 5: blind full autonomy after large validation and audit proof.

## Implementation Phases

1. Oracle diagnostics
   - Script: `oracle_diagnostics.py`
   - Measures `oracle@K` to decide whether misses are candidate-generation misses or ranking misses.

2. Gate rejection reporting
   - Script: `gate_rejection_report.py`
   - Explains why held tracks are not auto-save eligible.

3. Track-grouped validation
   - Updated: `train_candidate_chooser.py`, `train_groupwise_candidate_ranker.py`
   - Reports train/validation track counts and leakage checks.

4. Active learning queue
   - Script: `active_learning_queue.py`
   - Ranks held tracks by expected training value.
   - `web_review.py --queue models/active_review_queue.csv` reviews only that queue in order.

5. Structure map export
   - Script: `build_structure_maps.py`
   - Writes machine-readable per-bar `bar_lanes` with energy, groove, bass,
     drum density, vocal, novelty, and section labels.
   - This is the automation equivalent of a simplified waveform; models should
     consume these lanes rather than waveform screenshots.

6. Auto-save verifier
   - Module: `drop_aligner/auto_verifier.py`
   - Script: `train_auto_verifier.py`
   - Learns whether a selected marker is safe to auto-save.

7. Tuned auto gate
   - Script: `tune_auto_gate.py`
   - Searches held-out calibration thresholds for safe, balanced, and aggressive modes.

8. Autonomous run modes
   - Script: `auto_run.py`
   - Wraps existing reanalysis/autosave behavior with safe, balanced, aggressive, and audit modes.

9. Reporting
   - Updated: `pilot_report.py`
   - Adds oracle diagnostics, gate rejections, auto verifier metrics, and tuned gate coverage.

## Normal Workflow

```bash
python3 oracle_diagnostics.py \
  --corrections models/multistem_training_corrections.jsonl \
  --batch-summary ~/Desktop/MUSIC/STEMS/drop_batch_summary.csv

python3 gate_rejection_report.py ~/Desktop/MUSIC/STEMS/drop_batch_summary.csv

python3 active_learning_queue.py ~/Desktop/MUSIC/STEMS/drop_batch_summary.csv

python3 web_review.py --queue models/active_review_queue.csv
```

Build structure maps for the unreviewed library:

```bash
python3 build_structure_maps.py ~/Desktop/MUSIC/STEMS/drop_batch_summary.csv \
  --remaining-only \
  --workers 4 \
  --output-dir models/structure_maps
```

Rebuild gate and active-review reports from the real remaining-library payload:

```bash
python3 gate_rejection_report.py ~/Desktop/MUSIC/STEMS/drop_batch_summary.csv \
  --shadow-report eval_reports/remaining_reanalysis_after_retrain_20260603_091900_parallel_payload.json

python3 active_learning_queue.py ~/Desktop/MUSIC/STEMS/drop_batch_summary.csv
```

After more human reviews:

```bash
python3 train_multistem_drop_judge.py
python3 train_groupwise_candidate_ranker.py
python3 train_auto_verifier.py
python3 tune_auto_gate.py
python3 pilot_report.py --batch-summary ~/Desktop/MUSIC/STEMS/drop_batch_summary.csv
```

Dry-run autonomous save behavior before applying:

```bash
python3 auto_run.py ~/Desktop/MUSIC/STEMS/drop_batch_summary.csv --mode safe --dry-run --force
```

Apply only safe auto-saves with backups and ALS verification:

```bash
python3 auto_run.py ~/Desktop/MUSIC/STEMS/drop_batch_summary.csv --mode safe --force
```

## Safety Rules

- Do not train on auto-saved markers as truth unless later human-approved.
- Do not lower thresholds without calibration.
- Validate by track, not candidate row.
- Keep safe mode strict.
- Write held-for-review reasons for every rejected track.
- Back up ALS/CSV/JSON before applying saves.
- Use audit mode before increasing balanced/aggressive coverage.
