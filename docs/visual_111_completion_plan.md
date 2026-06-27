# Visual 1.1.1 Completion Plan

The target is not "better auto placement." The target is that every track gets
a DJ-usable Ableton `1.1.1` marker without manual correction. A marker is only
correct when it is both the visible front edge of a definitive drum/bass drop
body and on the BPM-derived musical one.

## Non-Negotiable Done Bar

Do not call this complete until all of these are true on the available library
and reviewed-history corpus:

1. `visual_first_marker(...)` returns a marker for every processable drums stem.
2. Every returned marker passes the visual audit with no `replace` or `review`
   status.
3. Every returned marker passes the BPM/on-one audit using the BPM from the
   filename or inferred beatgrid.
4. Every human-reviewed correction and skipped visual section has either been
   matched by the detector or converted into a regression fixture.
5. The broad reviewed-history evaluation has zero wrong-section misses and no
   outliers above the accepted millisecond tolerance for already-reviewed
   ground truth.
6. The full review queue/summary audit has zero suspects before any batch
   auto-save is allowed.

If any one of these fails, the detector is not done. The next step is to add a
specific regression and patch the visual detector rule that failed.

## Validation Loop

Run this loop repeatedly until the done bar is met:

1. Evaluate against all human-reviewed `1.1.1` corrections:

   ```bash
   venv/bin/python evaluate_visual_first.py --workers 4 --progress-every 25 \
     --output-json models/visual_first_audit_current.json \
     --output-csv models/visual_first_audit_current_misses.csv
   ```

2. Audit every track in the current review summary:

   ```bash
   venv/bin/python audit_visual_placements.py \
     /Users/alirahimlou/Desktop/MUSIC/STEMS/drop_batch_summary.csv \
     --scope all --workers 4 --progress-every 25 \
     --output-json models/visual_placement_audit_all_current.json \
     --output-csv models/visual_placement_audit_all_current_suspects.csv
   ```

3. Inspect the largest misses and suspects. Classify each as one of:

   - intro selected instead of drop
   - buildup/fake drop selected
   - breakdown selected
   - late hit inside the same thick body
   - blank/grid-only marker
   - off-one grid phase
   - first drop ambiguous and later definitive drop required
   - missing/unreadable audio or missing ground truth

4. Add or update a regression for that failure family in the visual-first test
   suite before changing detector behavior.
5. Patch only the visual detector path:

   - `drop_aligner/visual_first.py`
   - `drop_aligner/visual_drop_v2.py`
   - visual-first tests
   - visual detector docs

6. Re-run the focused tests for the changed area:

   ```bash
   venv/bin/python -m pytest drop_aligner/tests/test_visual_first.py::test_name_here -q
   ```

   Do not run the full visual-first file as one monolithic pytest command inside
   Codex; the real-audio examples can run for several minutes and make the
   session unstable. Use the bounded runner instead:

   ```bash
   venv/bin/python run_visual_first_regressions.py --chunk-size 10 --timeout-sec 120
   ```

7. Re-run the broad evaluation and placement audit. Repeat until the reports
   contain zero actionable misses/suspects.

## Auto-Save Gate

Batch auto-save is allowed only when the detector result has passed both
independent checks:

1. Visual check: the selected marker is at the first front edge of a definitive
   sustained drum/bass drop body, with visible waveform energy under the marker.
2. Grid check: the selected marker is on the returning one for the BPM clock.

If the detector cannot prove both checks, it must not silently write an approved
`1.1.1`. It should produce a suspect report that becomes the next regression.

## Reporting Requirement

Each iteration must report:

- number of tracks evaluated
- number of processable successes and failures
- number of wrong-section misses
- number of audit suspects
- worst remaining failure families
- exact files changed
- exact tests/audits run

Only after the reports show no remaining actionable failures should the system
be described as delivered.
