# Rhythm Algorithm Research Roadmap

Goal: make beat/downbeat/drop timing DJ-usable at millisecond precision across genres.

## What The Literature Says

- Transformer beat trackers can beat older DBN-heavy systems, but continuity failures still matter for hard genres. See Beat This: https://arxiv.org/abs/2407.21658
- Online systems still benefit from probabilistic sequence inference. BeatNet combines CRNN activations with particle filters for joint beat/downbeat/meter tracking: https://arxiv.org/abs/2108.03576
- Demixed inputs improve beat/downbeat tracking because stems expose drum, bass, harmonic, and vocal structure separately. See Beat Transformer: https://arxiv.org/abs/2209.07140
- Joint metrical plus structure learning is valuable because downbeats and drops are structural events, not isolated onsets. See All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio: https://arxiv.org/abs/2307.16425
- Drum-aware source-separated ensembles improve robustness on material with varying drum presence: https://arxiv.org/abs/2106.08685
- EDM cue/drop estimation can be framed as phrase-aligned object detection over spectrogram-like features. See CUE-DETR: https://arxiv.org/abs/2407.06823
- Standard beat tracking evaluation usually uses broad windows such as 70 ms and continuity metrics, which is not enough for DJ launch timing. We need 5/10/20 ms reporting and signed bias tracking in addition to 70 ms compatibility.

## Implementation Direction

1. Keep an ensemble of specialized providers: Beat This, BeatNet, madmom, stem ensemble, TrackOrganizer grid, and librosa fallback.
2. Measure every provider and final grid against reference beats/downbeats with strict 5/10/20 ms metrics, signed bias, and continuity.
3. Learn provider weights from real corrections and benchmark manifests instead of hardcoding trust.
4. Use Demucs stems twice: provider ensemble on stems for beat/downbeat tracking, and stem-aware micro-refinement for final attack alignment.
5. Treat first-drop detection as structural retrieval: phrase position, low-to-high energy transition, bass/drum re-entry, spectral flux, and cue prior agreement.
6. Build genre-specific calibration from references: House/Techno/DnB/Hip-Hop/acoustic material should be allowed different provider weights, fusion radii, and micro-refine windows.

## Current Code Changes

- `rhythm_engine.evaluate` now reports signed `median_error_ms` and `mean_error_ms`.
- `rhythm_engine.benchmark` can write an aggregate summary with final/provider/downbeat reports, genre reports, provider ranking, learned provider weights, and tuning recommendations.
- Project paths are centralized in `project_config.py`, so benchmark and automation runs can move between laptop, Docker, and CI without source edits.

## Next Hardening Targets

- Build a curated reference manifest from Rekordbox/Ableton/manual corrections with beat, downbeat, first-drop, genre, and stem availability columns.
- Run nightly parameter sweeps for `fusion_radius_ms`, `downbeat_fusion_radius_ms`, `fusion_min_*_gap_ratio`, and `micro_refine_window_ms`; pick configs by strict ms score, not only F1 at 70 ms.
- Train a cue/drop ranker using CUE-DETR-inspired region proposals plus our existing candidate features and correction logs.
- Add genre-conditioned provider weights and publish benchmark summaries for every release.
