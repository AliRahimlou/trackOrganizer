# Drop finder eval loop and 2026-08-01 baseline

The goal for every track: find the FIRST drop and anchor it exactly like the manual
Ableton workflow — warp marker sample-accurate on the drop's first bass/kick impact,
"Set 1.1.1 Here", identical anchor on all 3 stems (drums/inst/vocals), intro in
negative bars. `verify_als` now enforces the all-stems anchor contract
(`all_clips_have_drop_anchor` / `all_clips_share_drop_anchor`).

Ali's narrated rules (screen recording 2026-07-31, transcript in
`artifacts/screen_recordings/screen_recording_2026-07-31_224439_audio_transcript.txt`):
read intro/build-up/drop sections from the whole waveform; take the *beginning of
the first drop*; base it off the drums stem; zoom in until you see the waveform
"change from one type to the next" — that boundary is the drop; place 1.1.1 "as
close to the zero line as possible" (zero-crossing at the texture change); with all
3 stems selected so drums/inst/vocals share the exact same 1.1.1. Open experiment:
Stage B computes a zero-crossing diagnostic but does not move the marker — measure
whether snapping (≤1.5 ms) increases agreement with human picks in the ≤2 ms cohort.

## The instrument

```bash
./venv/bin/python visual_first_scorecard.py --workers 6
```

Runs the production Stage A + Stage B path (`visual_first_marker` →
`build_visual_first_als_anchor`) over every human-verified pick in
`models/multistem_training_corrections.jsonl` and
`models/post_reset_human_review_truth_59.jsonl` (later file wins per drums path)
and writes `models/scorecards/visual_first_scorecard_<stamp>.{json,csv}` with the
ms-error CDF, hold rate, tempo-band/selector splits, and worst offenders.

Run it before and after every detector or model change. Judge changes by this
scorecard plus "no NEW pytest failures"; absolute suite green is not the bar because
~100 `test_visual_first.py` per-track pins are aspirational.

## Baseline (old selector, 375 truth tracks)

- 21.6% holds (mostly `visual_marker_not_on_independent_one`)
- Of scored passes: 37% within 2 ms, but 43% in the wrong SECTION (>1 s off)
- Wrong-section direction 2:1 late: the cascade prefers a later/heavier body over
  the first drop
- Precision is solved; SELECTION is the failure mode

## Candidate coverage (atlas rebuild, 530 tracks)

`build_visual_candidate_atlas.py` over both correction logs:

- a candidate within 50 ms of the human pick exists for **80%** of tracks
- within 250 ms for **99%**

Candidate generation is essentially solved. Everything above selection strength is
ranking, guard passes, and grid phase.

## Retrained selector (2026-08-01)

`train_visual_candidate_selector.py --atlas-jsonl models/visual_candidate_atlas_20260801.jsonl`
(20% track holdout): top-pick within 50 ms went 34% → **54%** out-of-sample.

In production (full cascade, scorecard): within-2 ms 37.1→39.5%, within-90 ms
44.6→48.1%, median 419→100 ms, wrong-section 126→115, transitions +11 fixed / −2
broke. Old model kept at `models/visual_candidate_selector_pre20260801.pkl.bak`.

The production gain is much smaller than the offline gain because the learned
selector only locks at score ≥ 0.500 and ~40 guard/repair passes can still override
it. That gap is the roadmap.

## Round 2 (2026-08-01, second pass)

- `drop_aligner/energy_sections.py`: Ali's first-biggest-boost rule as a
  whole-track prior (66% within 1 bar of human picks standalone; opposite
  miss-bias to the cascade). Wired as selector feature
  `visual_first_top_boost_align` (kept inert — no holdout lift) and as the
  section gate for grid-phase recovery.
- Stage-B grid-phase recovery (`als_anchor._attempt_local_grid_phase_recovery`)
  with phase_error + micro_drift branches, gated on the energy prior.
  Truth-set: holds 84 -> 67 with zero regressions to good passes.
- FULLV2 set: 1000/1298 tracks placed (932 in V1), 295 holds, structural
  verification 1000/1000 identical 3-stem anchors, 803 anchors on a drums
  transient within 3 ms. 68 recovered tracks are tiered MEDIUM+ in the drums
  verification CSV for spot-checking.

## Next-round roadmap (highest leverage first)

1. **Trust the selector more**: when the model's top score is high and the pick
   passes Boom+GUI+audit proofs, stop letting later "stronger body" repair passes
   override it — the 2:1-late wrong sections come mostly from those passes.
2. **Hold recovery**: 70/81 holds are `visual_marker_not_on_independent_one` —
   iterate the Stage-B window / grid-phase reconciliation when grid confidence is
   low instead of hard-failing at −8 ms.
3. **Dense-mix Stage B**: kick-band onset scoring instead of broadband
   rise/(after+before) so busy pre-context stops blocking sample refinement.
4. **Fast-tempo path**: remove the `bar_sec < 1.35 s` bail-out (kills DnB) and make
   crest/attack windows tempo-relative.
5. Portable regression corpus (short excerpts) so the contract runs off this Mac.
