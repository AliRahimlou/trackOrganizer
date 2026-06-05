# Drop Detector Audit

Date: 2026-05-05

## Executive Summary

The current system is not ready for full autonomous drop placement yet. It has the right building blocks, but the active review queue is not using the strongest evidence:

- Current summary: 1301 processed tracks.
- Human reviewed: 125 tracks, all LOW confidence.
- Current model: trained on 125 correction rows / 1250 candidate rows.
- Current evaluation against reviewed rows: median error 0.323s, mean error 3.026s, 25ms accuracy 36.0%, 100ms accuracy 40.8%.
- Model closest-candidate accuracy: 57.6%.
- Handcrafted closest-candidate accuracy: 34.4%.
- Selected DrumPrint nonzero count: 0 / 1301.
- Selected MicroSnap nonzero count: 0 / 1301.

So the trained model is helping candidate choice, but it is still choosing among candidates that lack DrumPrint and MicroSnap features. The app cannot behave like a DJ until the production batch is regenerated with those features and the model is retrained on approved corrections from that feature-rich run.

## Fix Applied During Audit

The web GUI had a stale-rerank problem: retraining updated `models/drop_ranker.pkl`, but the running review page continued reading old `drop_batch_summary.csv` and old candidate JSON selections. That is why corrected training could improve the model while the visible AI marker still looked wrong.

Added:

- `drop_aligner/summary_rerank.py`
- `web_review.py` integration after retrain

Behavior:

- After a successful retrain, the app re-ranks existing candidate JSON files with the current production model.
- It updates `drop_batch_summary.csv`.
- It reloads the in-memory review queue.
- It does not re-run DSP.
- It does not train during comparison.
- It does not modify ALS files.

One manual rerank was run immediately:

- Rows scanned: 1301
- Candidate JSON updated: 1301
- Displayed AI times changed: 579
- Backup summary: `/Users/alirahimlou/Desktop/MUSIC/STEMS/drop_batch_summary.before_rerank_20260505_004654.csv`

The review server was restarted at `http://127.0.0.1:7860/`.

## Full-Groove Transition Layer Added

The proposed detector upgrade is effective and has now been implemented as an
additive ranking layer, not a replacement for the existing DSP detector.

Added:

- `drop_aligner/groove.py`
- spectral-flux feature extraction in `drop_aligner/detector.py`
- candidate features:
  - `drum_onset_spike`
  - `rms_jump`
  - `spectral_flux_peak`
  - `pre_drop_contrast`
  - `immediate_groove_start_score`
  - `groove_stability`
  - `sustained_full_groove_score`
- ranker input support with old-row fallback to `0.0`
- batch CSV columns for full-groove score, immediate groove, stability, and contrast
- web review and terminal review display for these values
- auto-accept gating that uses full-groove evidence when present
- debug plot spectral-flux overlay
- MicroSnap now considers nearby Ableton `.asd` transient markers as exact
  anchor candidates when they sit at the quiet pre-attack boundary

The purpose is to identify the first point where the drums become a sustained
section/groove, while penalizing isolated impacts and pre-drop accents that do
not start the groove immediately.

## Main Findings

### 1. DrumPrint and MicroSnap exist, but the current queue is not using them

The UI displays DrumPrint and MicroSnap fields, and the code has `drop_aligner/drumprint.py` and `drop_aligner/microalign.py`. But the current batch artifacts have zero selected DrumPrint and zero selected MicroSnap values. That means the model is mostly learning from:

- transient strength
- low-end jump
- post-drop density
- pre/post ratio
- rhythmic consistency
- snap offset
- raw timestamp

This is not enough for full autonomy. It can pick plausible impact points, but it does not yet know enough about repeated drum grooves or sample-level attack placement.

### 2. The ranker is too dependent on raw timestamp

Current production model feature importance is dominated by `timestamp`:

- `timestamp`: about 0.74
- `rhythmic_consistency`: about 0.08
- `confidence_score`: about 0.05
- `low_end_jump`: about 0.04
- DrumPrint and MicroSnap features: 0.0

This is why the model can improve average candidate choice while still making musically strange decisions. Raw timestamp can encode library habits, but it is not a robust musical reason.

### 3. The correction set is still too narrow

All 125 reviewed rows are LOW confidence. That is useful for fixing hard cases, but it gives the model no direct examples of:

- HIGH-confidence correct picks
- MEDIUM-confidence near misses
- HIGH-confidence regressions to avoid

The promotion gate reports LOW-tier accuracy only because the correction set only contains LOW rows.

### 4. Evaluation is useful but still optimistic

`evaluate_ranker.py` evaluates on the same correction rows used for training. That is good for smoke testing, but not enough to prove autonomy. The next evaluation should be group-held-out by track so the model is tested on songs it did not train on.

### 5. Older code contains useful staged logic that is not in the active web path

`FIRST_DOWNBEAT_ARCHITECTURE.md` and `first_downbeat_detector.py` already describe a stronger staged design:

1. rough region
2. Ableton marker adapter
3. local event parser
4. exact marker chooser
5. self-check
6. final arbitration

The active `drop_aligner/detector.py` path does not currently use most of that staged Ableton `.asd` / local-event machinery. Either merge the valuable pieces into `drop_aligner`, or explicitly retire the older path so the project has one production detector.

## Research Notes

The best path is not “one bigger transient detector.” The literature and library docs point to a layered system:

- Use percussive separation before onset/drop logic. Librosa exposes HPSS and returns harmonic/percussive components: https://librosa.org/doc/latest/generated/librosa.effects.hpss.html
- Use spectral-flux onset strength plus backtracking for attack starts. Librosa defines onset strength as positive spectral change and provides onset backtracking: https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html and https://librosa.org/doc/main/onset.html
- Use recurrence/self-similarity for repeated sections. Librosa recurrence matrices explicitly model frame repetition and affinity/self-similarity: https://librosa.org/doc/0.9.2/generated/librosa.segment.recurrence_matrix.html
- Use downbeat tracking, not just beat or transient tracking. Madmom’s RNN/DBN downbeat processor returns beat positions plus beat number inside the bar: https://madmom.readthedocs.io/en/v0.16/modules/features/downbeats.html
- Essentia gives a second rhythm/beat implementation with BPM, ticks, confidence, and beat intervals: https://essentia.upf.edu/reference/std_RhythmExtractor2013.html
- For onset features, Essentia documents HFC, complex, flux, melflux, and RMS onset detection functions: https://essentia.upf.edu/documentation/reference/streaming_OnsetDetection.html
- DrumPrint’s fingerprint idea matches Wang’s Shazam-style peak constellation concept: time-frequency landmark hashes are robust, sparse, and temporally localized: https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf
- Dynamic-programming beat tracking is the right mental model for enforcing tempo consistency instead of chasing isolated peaks: https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf

## Recommended Autonomous Architecture

### Stage A: Candidate Region

Use `drop_aligner/detector.py` as the main rough candidate generator, but add two required signals before trusting the result:

- DrumPrint section evidence: stable repeated drum landmarks after the marker.
- Self-similarity novelty: boundary between pre-drop and post-drop sections.

### Stage B: Musical Bar Alignment

Add a bar/downbeat layer:

- Use madmom if installed.
- Use Essentia if installed.
- Fall back to librosa beat tracking.
- Score candidates higher when they land on likely beat 1 of a 4-beat bar.

### Stage C: Exact Marker

Use MicroSnap for sample-level placement:

- Search around the selected region.
- Backtrack to attack start.
- Prefer clean zero crossing when confidence is high.
- Never move a marker a large distance without marking review needed.

### Stage D: Auto-Approval Gate

Only auto-approve when all evidence agrees:

- HIGH confidence, or strong MEDIUM.
- MicroSnap confidence high.
- DrumPrint pattern score nonzero and decent.
- Fake-hit penalty low.
- Candidate margin strong.
- Model and handcrafted are not strongly disagreeing.

Everything else stays in the review UI.

## Next Work

1. Re-run the batch with real DrumPrint and MicroSnap features:

```bash
python3 batch.py "/Users/alirahimlou/Desktop/MUSIC/STEMS" \
  --template "alsFiles/128.als" \
  --recursive \
  --stem-role drums \
  --strict-stem-set \
  --debug-candidates \
  --use-drumprint \
  --microalign \
  --no-hpss \
  --workers 4 \
  --force
```

2. Review a balanced set, not only LOW:

```bash
python3 web_review.py drop_batch_summary.csv \
  --template "alsFiles/128.als" \
  --review-medium-and-low \
  --regenerate-als-on-correction \
  --open-browser
```

3. Train and evaluate after 25-50 feature-rich approvals/corrections.

4. Run DrumPrint and MicroSnap A/B reports.

5. Add group-held-out evaluation before any default auto-approval expansion.

## Autonomy Decision Rule

Do not let the AI fully replace manual judgment until a pilot report shows:

- reviewed tracks: at least 50
- all feature-rich with DrumPrint and MicroSnap
- 25ms accuracy materially better than current 36%
- fake-hit rescues greater than fake-hit regressions
- HIGH-confidence reviewed tracks do not regress
- ALS validation pass rate remains 100%

Until then:

- AI can suggest and MicroSnap exact placement.
- Human approves LOW-confidence musical decisions.
- The model learns only from approved corrections, not from its own guesses.
