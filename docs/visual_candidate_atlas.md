# Visual Candidate Atlas

This project uses visual-first waveform review to place Ableton `1.1.1` on the
true drop impact. The detector must satisfy two checks before an accepted marker
is trusted:

1. The marker lands on the visible start of the sustained drums/bass drop body.
2. The marker is on the musical one for the title BPM or corrected beatgrid.

The atlas is the training map for improving that behavior from reviewed tracks.
It compares the current detector marker against every visual waveform candidate
generated for each human-reviewed track.

## Current Atlas

Built with:

```bash
venv/bin/python build_visual_candidate_atlas.py \
  --workers 4 \
  --progress-every 50 \
  --output-jsonl models/visual_candidate_atlas.jsonl \
  --output-csv models/visual_candidate_atlas_summary.csv
```

Outputs:

- `models/visual_candidate_atlas.jsonl`: one row per visual candidate.
- `models/visual_candidate_atlas_summary.csv`: one row per reviewed track.
- `artifacts/visual_candidate_atlas/miss_first_60/`: rendered waveform PNGs for
  the worst current detector misses.

Initial atlas result over 510 locally available human-reviewed tracks:

- Current detector within 50 ms: 274 / 510.
- Best available visual candidate within 50 ms: 372 / 510.
- Best available visual candidate within 250 ms: 399 / 510.
- Visual candidate rows: 5,961.

Latest expanded visual atlas result:

- Candidate rows: 162,760.
- Best available visual candidate within 50 ms: 452 / 510.
- Best available visual candidate within 250 ms: 506 / 510.

Latest gated runtime detector result:

- Detector within 50 ms: 449 / 510.
- Detector within 250 ms: 479 / 510.
- Wrong-section misses: 18.
- Median error: 0.82 ms.
- Report: `models/visual_first_audit_after_visual_selector_lock.json`.

## Interpretation

The expanded visual scan often finds a candidate close to the reviewed marker,
and the gated selector now picks most of those candidates without breaking the
saved GUI examples. Candidate selection remains the first bottleneck for future
tracks because held-out validation is still weaker than the in-sample reviewed
set.

The candidate generator remains the second bottleneck: 58 of the 510 reviewed
tracks still have no generated visual candidate within 50 ms of the human
marker, although only 4 are farther than 250 ms. Those cases need finer
sample-level visual boundary generation before a selector can reach 510 / 510.

## Next Algorithm Work

The next detector iteration should not widen hand-written replacement gates.
That already proved unsafe because bigger later waveform sections can override
reviewed-good early drops.

The next iteration should:

1. Train a groupwise visual selector from `models/visual_candidate_atlas.jsonl`.
2. Add candidate-generation fixes for tracks where the oracle candidate is
   farther than 250 ms from the reviewed marker.
3. Gate auto-accept on both selector confidence and audit pass.
4. Keep `later_definitive_drop_available`, `earlier_phrase_body_edge_available`,
   and `ambiguous_visual_drop_evidence` as review/training signals until the
   trained selector proves those moves against held-out human markers.

Confidence should mean the marker passed visual evidence, BPM-grid evidence, and
held-out atlas validation. It should not be set to 100% by display logic alone.
