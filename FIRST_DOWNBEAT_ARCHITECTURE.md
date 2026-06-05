# First Downbeat Architecture

This detector is intentionally staged. It should answer:

1. Where is the first real drop region?
2. Which local Ableton marker cluster is the true first drop event?
3. Which marker inside that event starts the sustained drop?
4. Does that marker still pass a local self-check?

The external output contract stays unchanged so `trackOrganizerAndAlsGen.py` can keep consuming:

```python
{
  "bpm": ...,
  "drums": {"downbeat_sample": ..., "downbeat_seconds": ..., "confidence": ...},
  "inst": {"aligned_sample": ..., "aligned_seconds": ...},
  "vocals": {"aligned_sample": ..., "aligned_seconds": ...},
  "debug": {...},
}
```

## Stage Layout

### 1. Rough Region

Lives in `first_downbeat_detector.py`:

- `_score_drums_candidates(...)`
- `_select_rough_custom_candidate(...)`
- `_build_rough_region_stage(...)`

This stage uses custom DSP and beat-grid scoring to find the approximate first real drop neighborhood. It is optimized for section-change evidence, not exact anchoring.

Key knobs:

- `rough_region_score_margin`
- `rough_region_conf_margin`
- `rough_region_min_density`
- `rough_region_near_best_margin`
- `rough_region_ref_density_margin`
- `rough_region_ref_lowend_margin`

Optional learned model support:

- `DOWNBEAT_ROUGH_REGION_MODEL`
- `DetectorOptions(rough_region_model_path=..., rough_region_model_blend=...)`

The detector can now blend a lightweight learned rough-region score into `_select_rough_custom_candidate(...)` without changing the external ALS-facing output contract.

### 2. Ableton Marker Adapter

Lives in `ableton_analysis_adapter.py`:

- `extract_ableton_onset_markers(...)`

This adapter harvests Ableton Live `.asd` transient markers and exposes them as normalized marker times in seconds/samples. These markers are the primary exact-anchor candidate set whenever `.asd` exists.

### 3. Local Event Parser

Lives in `first_downbeat_detector.py`:

- `_snap_to_rough_ableton_marker(...)`
- `_build_local_event_stage(...)`

This stage gathers nearby `.asd` markers around the rough region, groups them into local micro-events, and scores those events for true first-drop plausibility.

Important event cues:

- sustained post-event energy
- onset density after the event
- low-end growth
- pre/post contrast
- whether the event looks like preroll vs. a continuing musical section

Key knobs:

- `ableton_snap_search_beats`
- `ableton_cluster_max_spacing_beats`
- `ableton_cluster_window_beats`
- `ableton_event_prefer_earlier_margin`
- `ableton_event_min_sustain_norm`
- `ableton_event_min_density_norm`

### 4. Exact Marker Chooser

Lives in `first_downbeat_detector.py` inside `_snap_to_rough_ableton_marker(...)`.

This stage ranks the markers inside the chosen local event and selects the one that best represents the start of the first sustained drop event. It is not a naive earliest/nearest/strongest marker chooser.

Important marker cues:

- attack edge strength
- whether the attack already started before the marker
- whether the marker is inside the body of the hit
- sustained continuation after the marker
- dense-run followthrough
- gap to previous/next markers

Key knobs:

- `ableton_attack_start_min_score`
- `ableton_previous_marker_promotion_bonus`
- `ableton_later_in_cluster_penalty`
- `ableton_attack_started_penalty`
- `ableton_inside_body_penalty`
- `ableton_attack_start_reward`
- `ableton_attack_start_snap_beats`
- `ableton_attack_start_min_event_score`
- `ableton_attack_start_min_support`
- `ableton_attack_start_min_edge`
- `ableton_attack_start_max_inside_body`
- `ableton_attack_start_min_followthrough`

### 5. Self-Check

Lives in `first_downbeat_detector.py`:

- local self-check block inside `_snap_to_rough_ableton_marker(...)`
- `_build_self_check_stage(...)`

This validates that the chosen marker still looks like the start of a sustained event. If not, confidence is reduced and the `.asd` marker may be rejected as the final anchor.

Key knobs:

- `ableton_self_check_margin`
- `ableton_cluster_earliest_min_support`
- `ableton_cluster_max_post_attack_drift_beats`
- `ableton_cluster_max_early_pull_beats`

### 6. Final Arbitration

Lives in `first_downbeat_detector.py` inside `detect(...)`.

This is where the detector decides whether the final drums anchor should come from:

- the refined custom candidate
- the local Ableton event / exact marker
- legacy fallback

The current production intent is:

- custom DSP finds the rough region
- Ableton `.asd` usually supplies the exact marker
- fallback stays available when `.asd` is missing or the local event fails validation
- if an earlier `.asd` marker is only a tiny attack-start shift before the current anchor and it has strong event-start evidence, it can now override the safer later custom anchor

Key knobs:

- `ableton_exact_anchor_prefer_within_beats`
- `ableton_exact_anchor_min_event_score`
- `ableton_exact_anchor_min_support`
- `ableton_override_earlier_beats`
- `ableton_override_min_support`
- `ableton_override_conf_tolerance`

## Debug Output

`SongDownbeatResult["debug"]` now exposes staged debug:

- `stages.rough_region`
- `stages.local_event`
- `stages.self_check`
- `ableton_snap`

Plot rendering still lives in `_write_debug_plots(...)` and can show:

- waveform
- rough region
- nearby `.asd` markers
- local event boundaries
- chosen marker
- manual anchor overlay when available

If `matplotlib` is unavailable, a note file is written instead of failing silently.

## Evaluator / Training Export

`evaluate_downbeat_detector.py` can export:

- per-track rows via `--out-csv`
- per-marker candidate rows via `--out-candidate-csv`
- per-custom-candidate rough-region rows via `--out-rough-candidate-csv`

Candidate rows include:

- feature columns for each nearby `.asd` marker
- event/cluster metadata
- chosen vs. not chosen
- correctness labels against manual CH1 anchors
- stage reasons like `rough_region_reason`, `selected_event_reason`, `self_check_reason`

Rough-region rows include:

- one row per custom DSP candidate before `.asd` snapping
- structural feature columns used by the rough-region selector
- labels such as `candidate_is_manual_match_0p25beat`, `candidate_is_manual_match_0p50beat`, and `candidate_is_manual_match_1beat`
- `candidate_is_custom_reference`, `candidate_is_rough_choice`, and `candidate_is_final_prediction`

## Future Learned Ranker

Both the rough-region stage and exact-marker stage are trainable.

Current support:

- weighted rough-region scoring with optional learned blend
- weighted heuristic ranking
- optional JSON logistic model loaded via `DOWNBEAT_MARKER_RANK_MODEL`
- optional JSON logistic rough-region model loaded via `DOWNBEAT_ROUGH_REGION_MODEL`

Trainer:

- `train_downbeat_marker_ranker.py`
- `train_rough_region_ranker.py`

Intended future path:

1. export rough-region rows and nearby-marker rows from the evaluator
2. train a rough-region model on manual CH1 labels to improve first-drop neighborhood selection
3. train a local `.asd` marker ranker on the chosen event markers
4. load the resulting model JSON artifacts into the detector
5. keep ALS integration unchanged

## Fallback

If `.asd` is missing or parsing fails:

- the detector falls back to the current custom-only path
- downstream ALS integration remains unchanged
