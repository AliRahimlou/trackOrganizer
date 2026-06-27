# Visual 1.1.1 Simplification Audit

The production goal is simple: write `1.1.1` only when the marker has both
visible drop-body evidence and BPM-grid on-one evidence. Everything else should
hold for review instead of growing another rescue path.

## Keep Optimizing

- The visual/body proof contract.
- BPM on-one proof.
- Human-reviewed ground truth and regression seeds.
- Reference-preserving ALS writing and ALS/file-ref validation.
- Independent suspicious-marker audit.

## Stop Optimizing

- One-off rescue paths for ambiguous tracks. If a marker cannot prove visual
  body plus grid-on-one, hold it and turn it into a regression case.
- Full monolithic `test_visual_first.py` runs inside Codex. Use
  `run_visual_first_regressions.py` so slow real-audio examples run in bounded
  chunks and cannot orphan pytest processes.
- Draft combined-set outputs from the top-level automation. The wrapper now
  exposes one production path: build, validate, then suspicious audit.

## Removed

- Disabled `visual_drop_v2` production fallback code from
  `drop_aligner/visual_first.py`. The standalone v2 module and unit tests remain
  as research coverage, but the production visual-first marker path no longer
  imports, calls, or handoffs through it.
- Rough-unreferenced helpers in `drop_aligner/visual_first.py`:
  `_final_whole_track_gui_front_edge_cluster_repair`,
  `_unresolved_earlier_phrase_body_conflict`, and the unused risky-v2 selector
  override.

## Remaining Delete Candidates

- Public unsafe/draft paths outside the lower-level builder. The top-level
  `run_visual_111_automation.py` no longer forwards `--allow-partial` or
  `--allow-unsafe-audit`; keep those only for direct builder debugging.

## Simplified Operating Rule

Do not make the detector smarter by adding more exceptions until the failure is
classified. For each miss:

1. Add it to the failure inventory.
2. Confirm the true marker manually or from trusted review data.
3. Add a regression.
4. Make the smallest detector change.
5. Run focused tests plus bounded real-audio chunks.

If step 2 cannot prove a marker, the correct production behavior is `HOLD`, not
another auto-repair.
