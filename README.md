# trackOrganizer

Drop-aligner docs live in [drop_aligner/README.md](drop_aligner/README.md).

The active production path is visual-first Boom-waveform alignment. It works
from the full drums-stem waveform, proves the marker against the sustained
drum/bass body, checks the BPM one, and blocks stale non-visual detector writes.

Visual-first `1.1.1` automation:

```bash
python3 run_visual_111_automation.py --force
```

Optional CuPy acceleration for compatible pure-array kernels:

```bash
pip install -r requirements-gpu-cuda12.txt
TRACKORGANIZER_ARRAY_BACKEND=auto python3 run_visual_111_automation.py --force
```

The GPU backend is intentionally optional. The detector still keeps NumPy arrays
at `librosa`, SciPy, scikit-learn, and PyTorch boundaries, and falls back to
NumPy when CuPy or a CUDA device is unavailable. Set
`TRACKORGANIZER_ARRAY_BACKEND=numpy` to force CPU behavior.

This runs the same full-drums-waveform path used in manual review: locate the
drop body visually, refine to the first launch edge, require the BPM-grid one,
then validate the generated ALS/report with fail-closed production checks. Holds
are written to the generated failure and audit CSVs instead of being approved.

Legacy full automation:

```bash
./run_full_automation.sh
```

This launcher uses the project `venv`, enables the fused drop audit, reads
Mixed In Key cue tags and Rekordbox priors when available, and only lets the
fusion layer override the old detector when the evidence gate marks the
candidate safe. Close calls are written to each track's `drop_fusion_audit.json`
and held to the existing detector path instead of blindly rewriting 1.1.1.

Recommended production flow:

```bash
python3 build_fresh_visual_first_library_set.py --workers 4 --force
python3 validate_visual_first_production.py /path/to/visual_first_report.json --out-dir /path/to/output-dir --workers 4
python3 web_review.py /path/to/visual_first_summary.csv --template "alsFiles/128.als" --visual-first --no-open-browser
```

`validate_visual_first_production.py` reruns the current visual-first detector by
default and requires those fresh markers to match the saved production report.
Use `--no-rerun-detector` only for a quick persisted-payload audit, not for a
production-ready library gate.

`batch.py`, `main.py`, `auto_review.py`, `apply_auto_place_initial.py`, and
`reanalyze_remaining_hard_cases.py` are legacy non-visual write paths. They are
blocked by default and require `--allow-legacy-detector-write` or
`TRACK_ORGANIZER_ALLOW_LEGACY_WRITES=1` for an intentional experiment.
