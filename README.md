# trackOrganizer

Drop-aligner docs live in [drop_aligner/README.md](drop_aligner/README.md).

The active detector combines the original DSP score with DrumPrint, MicroSnap,
and a sustained full-groove transition layer so candidates are ranked by the
start of a real drum section, not just the loudest transient.

Fully automated library run:

```bash
./run_full_automation.sh
```

This launcher uses the project `venv`, enables the fused drop audit, reads
Mixed In Key cue tags and Rekordbox priors when available, and only lets the
fusion layer override the old detector when the evidence gate marks the
candidate safe. Close calls are written to each track's `drop_fusion_audit.json`
and held to the existing detector path instead of blindly rewriting 1.1.1.

Recommended pilot flow:

```bash
python3 batch.py "/path/to/edm/library" --template "CH1.als" --recursive --debug-candidates --use-drumprint --microalign --workers 4 --force
python3 web_review.py drop_batch_summary.csv --template "CH1.als" --review-medium-and-low --regenerate-als-on-correction --open-browser
python3 review.py drop_batch_summary.csv --retrain
python3 evaluate_ranker.py --corrections drop_corrections.jsonl --model models/drop_ranker.pkl
python3 compare_drumprint.py drop_batch_summary.csv --template "CH1.als" --corrections drop_corrections.jsonl --workers 4
python3 compare_microalign.py drop_batch_summary.csv --template "CH1.als" --corrections drop_corrections.jsonl --workers 4
python3 pilot_report.py --batch-summary drop_batch_summary.csv --corrections drop_corrections.jsonl --evaluation models/evaluation_report.json --drumprint-ablation models/drumprint_eval/drumprint_ablation_report.json --microalign-ablation models/microalign_eval/microalign_ablation_report.json
```

Do not enable conservative auto-review broadly until MicroSnap improves median
error, 10ms accuracy, or 25ms accuracy without meaningful HIGH-confidence
regressions.
