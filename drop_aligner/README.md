# Drop Aligner

Production-oriented first-drop alignment for Ableton Live templates.

The detector treats cue points as search regions, not exact anchors. It finds a
sustained first-drop impact in the drums/audio, snaps the result back to the true
attack start and nearest preceding zero crossing, then writes a new `.als` from a
provided template.

## Usage

```bash
python3 main.py "/path/to/track.flac" --template "/path/to/CH1.als"
```

Optional cue regions:

```bash
python3 main.py "track.flac" --template "CH1.als" --cue 30.0 --cue 60.0
```

Debug outputs:

```bash
python3 main.py "track.flac" --template "CH1.als" \
  --debug-candidates
```

This writes `track_drop_candidates.json` and `track_drop_debug.png` beside the
audio. The JSON includes the top 10 ranked candidates, final AI pick, selected
candidate, snap offsets, energy/density/rhythm features, full-groove transition
features, and rejection/selection reasons. It also includes `confidence_tier`:
`HIGH`, `MEDIUM`, or `LOW`.

The base detector now includes a first sustained full-groove layer. For each
candidate, it compares the pre-drop region against the first bars after the
candidate and scores low-end jump, drum onset spike, RMS jump, spectral-flux
novelty, pre-drop contrast, immediate groove start, and bar-to-bar groove
stability. This is meant to prefer the first marker where the drums become a
real section, not a one-off transient.

It also includes an EDM transition scan for buildup-to-drop cases. That scan
adds candidates where a low/less-bassy pre-drop window is followed by sustained
RMS, low-end, spectral-flux, and kick re-entry. Candidate JSON now includes
`buildup_low_energy_score`, `buildup_ramp_score`, `drop_impact_score`,
`kick_reentry_score`, and `buildup_drop_score`; these help avoid selecting
rising buildup hits before the real first drop.

For drums stems, the detector also adds a DrumPrint fingerprint layer by
default. It builds sparse percussive spectrogram landmarks and scores candidates
by sustained fingerprint density, novelty, post-drop pattern stability,
kick-pattern repeat, self-similarity boundary strength, later pattern matches,
and fake-hit penalty. This does not replace the DSP detector; it only adds a
ranking feature. Override it with:

```bash
python3 main.py "drums_128_track.wav" --template "CH1.als" --use-drumprint
python3 main.py "drums_128_track.wav" --template "CH1.als" --no-drumprint
```

Sample-level MicroSnap refinement can be enabled when you want the detector to
place the exact marker inside the selected region:

```bash
python3 main.py "drums_128_track.wav" --template "CH1.als" --microalign
```

MicroSnap uses the raw audio samples, not waveform screenshots. It analyzes a
small high-resolution window around the candidate, finds the clean drum attack,
backtracks to the attack boundary, and optionally snaps to a nearby zero
crossing. When an Ableton `.asd` analysis file exists beside the audio, MicroSnap
also considers Ableton's own transient markers as exact anchor candidates and
prefers one when it sits at the quiet pre-attack boundary before sustained
energy. It only moves the selected marker when confidence is high and the offset
is reasonable; otherwise the marker is left for review.

You can still override the full analysis JSON or plot path:

```bash
python3 main.py "track.flac" --template "CH1.als" \
  --analysis-json "track_DROP_ANALYSIS.json" \
  --plot "track_DROP_DEBUG.png"
```

Correction logging:

```bash
python3 main.py "track.flac" --template "CH1.als" \
  --debug-candidates \
  --user-pick 45.123
```

This appends a JSONL row containing the final AI pick, user pick, delta, top 10
candidates, closest candidate to the user pick, selected candidate, and feature
values for later ranking-model training.

Train Ali's candidate ranker:

```bash
python3 train_ranker.py --corrections drop_corrections.jsonl
```

This saves `models/drop_ranker.pkl`. When that file exists, the DSP detector
still finds candidates exactly as before, then the model re-ranks the top 10
candidates by predicted distance to Ali's preferred marker. If no model exists,
the detector falls back to the handcrafted score. Training also writes
`models/training_report.json` with accepted/corrected counts, average delta, and
model feature importances.

Batch folder processing:

```bash
python3 batch.py "/path/to/folder" \
  --template "CH1.als" \
  --recursive \
  --debug-candidates \
  --workers 4
```

Supported audio files are `.wav`, `.flac`, `.aiff`, `.aif`, and `.mp3`.
For each track, batch mode writes:

- `*_DROP_ALIGNED.als`
- `*_drop_candidates.json`
- `*_drop_debug.png` when `--debug-candidates` is set
- `drop_batch_summary.csv` in the scanned folder

Already processed tracks are skipped when the expected outputs exist. Use
`--force` to regenerate them.

Batch MicroSnap fields:

```bash
python3 batch.py "/path/to/folder" \
  --template "CH1.als" \
  --recursive \
  --stem-role drums \
  --use-drumprint \
  --microalign \
  --debug-candidates
```

When `--microalign` is enabled, candidate JSON includes `microalign` data and
the summary CSV includes `micro_confidence`, `snap_offset_ms`, and
`microaligned_time`.

Dry run:

```bash
python3 batch.py "/path/to/folder" --template "CH1.als" --recursive --dry-run
```

The summary CSV columns are `filename`, `detected_drop_time`, `confidence`,
`confidence_tier`, `selected_by`, `sustained_full_groove_score`,
`immediate_groove_start_score`, `groove_stability`, `pre_drop_contrast`,
`drumprint_pattern_score`, `fake_hit_penalty`, MicroSnap fields when enabled,
`output_als`, `candidates_json`, `debug_png`, `als_valid`,
`als_validation_error`, `status`, and `error`. A failed track records
`status=error` and does not stop the batch.

Verify an Ableton file:

```bash
python3 verify_als.py "track_DROP_ALIGNED.als" \
  --candidates-json "track_drop_candidates.json"
```

This checks gzip/XML validity, AudioClip and WarpMarkers presence, the
`SecTime=0` marker, exactly one non-zero `BeatTime=0` drop marker, monotonic
marker timing, audio file references, and candidate JSON time agreement. Use
`--allow-multiple` for templates that intentionally contain more than one drop
marker. Use `--json report.json` to write a machine-readable report.

Review batch results and build training data:

```bash
python3 review.py drop_batch_summary.csv
```

Review mode prioritizes `LOW`, then `MEDIUM`, then `HIGH` confidence tracks. For
each row it opens the debug PNG, prints the detected drop time, confidence tier,
selected method, and top 10 candidates, then waits for:

- `Enter` to accept the AI pick
- a timestamp like `1:03.250` to write a corrected user pick
- `s` to skip the track
- `q` to quit review

Accepted and corrected rows are appended to `drop_corrections.jsonl`. For
accepted rows, `user_pick` equals the AI pick. Correction rows include the track,
AI pick, user pick, delta, top 10 candidates, closest candidate to the user pick,
and `selected_by`.

Only review uncertain tracks:

```bash
python3 review.py drop_batch_summary.csv --review-low-only
python3 review.py drop_batch_summary.csv --review-medium-and-low
```

Review and retrain:

```bash
python3 review.py drop_batch_summary.csv --retrain
```

After review, this trains a candidate model at
`models/drop_ranker_candidate.pkl`, evaluates it against the current production
model at `models/drop_ranker.pkl`, and only promotes it if the quality gate
passes:

- mean absolute error improves, or 25ms accuracy improves
- LOW-confidence-tier accuracy does not regress

If promoted, the old model is backed up to `models/drop_ranker_previous.pkl`,
then the candidate replaces `models/drop_ranker.pkl`. The decision is written to
`models/promotion_report.json`.

Local browser review:

```bash
python3 web_review.py
```

By default this uses `/Users/alirahimlou/Desktop/MUSIC/STEMS/drop_batch_summary.csv`,
`alsFiles/128.als`, opens `http://127.0.0.1:7860`, regenerates corrected ALS
files, and retrains every 25 approvals/corrections. If the summary CSV is
missing, it runs the default drums-stem batch first. Use explicit arguments only
when reviewing a different folder or template:

```bash
python3 web_review.py "/path/to/drop_batch_summary.csv" --template "alsFiles/128.als"
```

The browser review shows one track at a time
with the debug PNG, interactive waveform/audio preview, AI marker, candidate
markers, full-groove/DrumPrint/MicroSnap metrics, and correction marker
placement. The waveform is rendered from backend tiles instead of full-track
browser decoding: normal zoom uses min/max peak tiles, deep zoom uses raw
samples, and extreme zoom shows sample points and zero crossings. Marker
placement and `+/-1 sample` nudging snap to the source sample rate. Actions:

- `YES, correct` logs an approval with `user_pick = ai_pick`
- `NO, place correct marker` lets you click the waveform and nudge by 1ms/10ms
- `AI REFINE MARKER` runs MicroSnap on the current marker and shows the refined marker
- `AI AUTO PLACE` runs MicroSnap on the top candidates and suggests a final marker
- `ACCEPT AI REFINED MARKER` logs your approval of that refined marker and rewrites/verifies the ALS
- `Export PNG` writes a high-resolution PNG of the current zoom window
- `SAVE CORRECTED ALS` logs the correction and rewrites/verifies the ALS when
  `--regenerate-als-on-correction` is set
- `SKIP` leaves the track unlogged
- `RETRAIN NOW` runs the same quality-gated candidate promotion flow as
  `review.py --retrain`

Progress is saved to `review_state.json` beside the batch summary. Audio stays
local; if `ffmpeg` is available the server makes a short WAV preview around the
detected marker for browser compatibility.

Visual-first review:

```bash
python3 web_review.py "/path/to/drop_batch_summary.csv" \
  --template "alsFiles/128.als" \
  --visual-first \
  --regenerate-als-on-correction
```

This opens each track on the full waveform instead of the AI marker window.
The waveform draws darker sustained-energy chunks so the first big visual block
is easier to spot. Double-click a block to zoom into it, repeat until the edge is
clear, place `1.1.1`, optionally run `AI REFINE`, then save the marker through
the normal ALS regeneration and verification path.

Automatic MicroSnap suggestions:

```bash
python3 auto_review.py drop_batch_summary.csv --template "CH1.als" \
  --mode conservative \
  --write-auto-log auto_marks.jsonl
```

Modes are `conservative`, `normal`, and `aggressive`. Automatic placements are
written to `auto_marks.jsonl`, not `drop_corrections.jsonl`. In conservative
mode, ALS regeneration only happens when the shared auto-accept gate passes.
Do not train on auto marks as if they were human labels; only your approved web
UI corrections belong in the supervised correction log.

Evaluate ranker performance:

```bash
python3 evaluate_ranker.py \
  --corrections drop_corrections.jsonl \
  --model models/drop_ranker.pkl
```

This writes `models/evaluation_report.json` and
`models/evaluation_report.csv`, and prints median/mean error, percent within
5/10/25/50/100ms, handcrafted-vs-model closest-candidate accuracy, and the worst
20 misses. The JSON report also includes accuracy and average delta by
confidence tier.

Evaluate DrumPrint before rolling it across the full library:

```bash
python3 compare_drumprint.py drop_batch_summary.csv --template "CH1.als" \
  --corrections drop_corrections.jsonl \
  --workers 4 \
  --output-dir models/drumprint_eval
```

Optional golden references:

```bash
python3 compare_drumprint.py drop_batch_summary.csv --template "CH1.als" \
  --golden golden_tracks.json
```

Evaluate MicroSnap before trusting automatic placement:

```bash
python3 compare_microalign.py drop_batch_summary.csv --template "CH1.als" \
  --corrections drop_corrections.jsonl \
  --workers 4 \
  --output-dir models/microalign_eval
```

This runs detection with `--no-microalign` and `--microalign`, compares both
against corrections or golden tracks, reports 1/5/10/25/50/100ms accuracy, and
counts conservative/normal/aggressive auto-accept eligibility. It does not train,
rewrite ALS files, or overwrite candidate JSON.

Recommended pilot workflow:

1. Run batch with DrumPrint and MicroSnap enabled for drums stems:
   `python3 batch.py "/path/to/folder" --template "CH1.als" --recursive --stem-role drums --use-drumprint --microalign --debug-candidates`
2. Review and correct 25-50 tracks with `python3 web_review.py`.
3. Train/promote the correction model with `python3 review.py drop_batch_summary.csv --retrain` or `RETRAIN NOW` in the browser review.
4. Run `evaluate_ranker.py` against `drop_corrections.jsonl`.
5. Run `compare_drumprint.py` and `compare_microalign.py` against corrections
   or golden tracks.
6. Summarize the pilot:

   ```bash
   python3 pilot_report.py \
     --batch-summary drop_batch_summary.csv \
     --corrections drop_corrections.jsonl \
     --evaluation models/evaluation_report.json \
     --drumprint-ablation models/drumprint_eval/drumprint_ablation_report.json \
     --microalign-ablation models/microalign_eval/microalign_ablation_report.json
   ```

7. Keep DrumPrint enabled by default only if the reports show median error
   improvement, 25ms accuracy improvement, or more fake-hit rescues than
   fake-hit regressions, with no significant regression on already-correct
   HIGH-confidence tracks.
8. Enable conservative auto-review only after MicroSnap improves median error,
   10ms accuracy, or 25ms accuracy, with rare worst regressions and rare snap
   offsets over 100ms.

The comparison writes:

- `models/drumprint_eval/drumprint_ablation_report.json`
- `models/drumprint_eval/drumprint_ablation_report.csv`
- `models/microalign_eval/microalign_ablation_report.json`
- `models/microalign_eval/microalign_ablation_report.csv`

It does not train, rewrite ALS files, or overwrite candidate JSON.

`pilot_report.py` writes `pilot_report.json` and `pilot_report.md`. It combines
batch results, human review corrections, ranker evaluation, DrumPrint A/B, and
MicroSnap A/B results into one readable summary with action recommendations. It
is reporting only: it does not train, rewrite ALS files, or overwrite model
files.

Project health check:

```bash
python3 doctor.py
```

This prints simple `PASS`, `WARN`, and `FAIL` checks for Python, required
packages, template readability, trained model presence, correction log presence,
and one recent generated ALS when available. It ends with a recommended next
action.
