#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from .build_gold_dataset import run_build_gold_dataset
from .evaluate import run_evaluate
from .extract_dataset import run_extract
from .features import run_features
from .infer import run_predict
from .train import run_train
from .utils import find_audio_files, parent_dir, write_json
from .write_als import run_validate_als, run_write_als


def _print_json(obj: Dict[str, object]) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=True))


def _cmd_extract(args) -> int:
    report = args.report.strip() if args.report else ""
    if not report:
        report = os.path.join(os.path.dirname(os.path.abspath(args.out)), "dataset_report.html")
    res = run_extract(
        als_dir=args.als_dir.strip() or None,
        als_paths=list(args.als or []),
        als_globs=list(args.als_glob or []),
        out_jsonl=args.out,
        include_unwarped=bool(args.include_unwarped),
        no_resolve_audio=bool(args.no_resolve_audio),
        report_html=report,
    )
    _print_json(res)
    return 0


def _cmd_features(args) -> int:
    res = run_features(
        dataset_jsonl=args.dataset,
        cache_dir=args.cache_dir,
        manifest_out=args.manifest,
        sr=int(args.sr),
        hop=int(args.hop),
        n_mels=int(args.n_mels),
        force=bool(args.force),
        limit=int(args.limit),
        workers=int(args.workers),
    )
    _print_json(res)
    return 0


def _cmd_train(args) -> int:
    res = run_train(
        dataset_jsonl=args.dataset,
        silver_dataset_jsonl=args.silver_dataset,
        feature_manifest_jsonl=args.features,
        candidates_dir=args.candidates_dir,
        out_model=args.out,
        out_metrics=args.metrics,
        use_madmom=not bool(args.no_madmom),
        force_candidates=bool(args.force_candidates),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_neg=int(args.max_neg),
        max_hard=int(args.max_hard),
        rank_weight=float(args.rank_weight),
        listwise_weight=float(args.listwise_weight),
        bce_weight=float(args.bce_weight),
        offset_weight=float(args.offset_weight),
        primary_weight=float(args.primary_weight),
        silver_weight=float(args.silver_weight),
        seed=int(args.seed),
        device=args.device,
        limit_tracks=int(args.limit_tracks),
        tensor_cache_capacity=int(args.tensor_cache_capacity),
        tracks_per_step=int(args.tracks_per_step),
        model_width=int(args.model_width),
        model_dropout=float(args.model_dropout),
        long_frames=int(args.long_frames),
        short_frames=int(args.short_frames),
        max_candidates=int(args.max_candidates),
        candidate_tol_sec=float(args.candidate_tol_sec),
        require_candidate_hit=not bool(args.no_require_candidate_hit),
    )
    _print_json(res)
    return 0


def _cmd_predict(args) -> int:
    res = run_predict(
        audio_path=args.audio,
        model_path=args.model,
        out_json=args.out.strip() or None,
        device=args.device,
        use_madmom=not bool(args.no_madmom),
        bpm_override=args.bpm if args.bpm > 0 else None,
        review_threshold=float(args.review_threshold),
    )
    _print_json(res)
    return 0


def _cmd_write_als(args) -> int:
    res = run_write_als(
        template_als=args.template,
        audio_path=args.audio,
        predicted_json=args.pred.strip() or None,
        out_als=args.out,
        predicted_sec=args.pred_sec,
        bpm_override=args.bpm,
        apply_to_all=bool(args.apply_to_all),
    )
    _print_json(res)
    return 0


def _cmd_validate_als(args) -> int:
    res = run_validate_als(args.als, expected_target_sec=args.expected_sec)
    _print_json(res)
    return 0 if bool(res.get("ok")) else 2


def _cmd_evaluate(args) -> int:
    res = run_evaluate(
        dataset_jsonl=args.dataset,
        model_path=args.model,
        out_json=args.out,
        use_madmom=not bool(args.no_madmom),
        device=args.device,
        review_threshold=float(args.review_threshold),
        limit=int(args.limit),
        candidate_tol_sec=float(args.candidate_tol_sec),
        debug_dir=(args.debug_dir.strip() or None),
        report_html=(args.report_html.strip() or None),
        seen_dataset_jsonl=(args.seen_dataset.strip() or None),
        split_from_model_path=(args.split_from_model.strip() or None),
        split_name=str(args.split_name or "test"),
    )
    _print_json(res.get("summary", {}))
    return 0


def _cmd_tier(args) -> int:
    res = run_build_gold_dataset(
        dataset_jsonl=args.dataset,
        out_gold=args.out_gold,
        out_silver=args.out_silver,
        out_bronze=args.out_bronze,
        out_report_json=args.report_json,
        out_report_html=args.report_html,
        feature_manifest_jsonl=args.features_manifest,
        drums_only=not bool(args.no_drums_only),
        min_gold_rows=int(args.min_gold_rows),
    )
    _print_json(res)
    return 0


def _cmd_batch(args) -> int:
    audio_files = find_audio_files(args.audio_dir)
    if args.limit and int(args.limit) > 0:
        audio_files = audio_files[: int(args.limit)]
    if not audio_files:
        raise RuntimeError(f"No audio files found under: {args.audio_dir}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ok = 0
    fail = 0
    rows: List[Dict[str, object]] = []

    for ap in audio_files:
        stem = os.path.splitext(os.path.basename(ap))[0]
        pred_out = os.path.join(out_dir, f"{stem}.predicted.json")
        als_out = os.path.join(out_dir, f"{stem}.warped.als")
        try:
            pred = run_predict(
                audio_path=ap,
                model_path=args.model,
                out_json=pred_out,
                device=args.device,
                use_madmom=not bool(args.no_madmom),
                bpm_override=args.bpm if args.bpm > 0 else None,
                review_threshold=float(args.review_threshold),
            )
            wr = run_write_als(
                template_als=args.template,
                audio_path=ap,
                predicted_json=pred_out,
                out_als=als_out,
                predicted_sec=None,
                bpm_override=args.bpm if args.bpm > 0 else None,
                apply_to_all=bool(args.apply_to_all),
            )
            row = {
                "audio_path": ap,
                "pred_json": pred_out,
                "output_als": als_out,
                "predicted_sec": float(pred.get("predicted_sec", 0.0)),
                "confidence": float(pred.get("confidence", 0.0)),
                "needs_manual_review": bool(pred.get("needs_manual_review", False)),
                "validation_ok": bool((wr.get("validation") or {}).get("ok", False)),
            }
            rows.append(row)
            ok += 1
        except Exception as e:
            rows.append({"audio_path": ap, "error": str(e)})
            fail += 1

    summary = {
        "ok": True,
        "total": int(len(audio_files)),
        "success": int(ok),
        "failed": int(fail),
        "out_dir": out_dir,
        "rows": rows,
    }

    out_manifest = os.path.join(out_dir, "batch_results.json")
    write_json(out_manifest, summary)
    summary["manifest"] = out_manifest
    _print_json({k: v for k, v in summary.items() if k != "rows"})
    return 0 if fail == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alsdrop", description="ALS-supervised Ableton 1.1.1 drop anchor learner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="Extract dataset from ALS files")
    p.add_argument("--als_dir", default="", help="Directory to recursively scan for .als")
    p.add_argument("--als", action="append", default=[], help="Explicit .als path (repeatable)")
    p.add_argument("--als-glob", action="append", default=[], help="Glob pattern for ALS files")
    p.add_argument("--out", default="alsdrop/data/dataset.jsonl")
    p.add_argument("--report", default="")
    p.add_argument("--include-unwarped", action="store_true")
    p.add_argument("--no-resolve-audio", action="store_true")
    p.set_defaults(func=_cmd_extract)

    p = sub.add_parser("features", help="Extract and cache features")
    p.add_argument("--dataset", default="alsdrop/data/dataset.jsonl")
    p.add_argument("--cache_dir", default="alsdrop/data/features")
    p.add_argument("--manifest", default="alsdrop/data/features_manifest.jsonl")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=96)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.set_defaults(func=_cmd_features)

    p = sub.add_parser("train", help="Train model")
    p.add_argument("--dataset", default="alsdrop/data/dataset.jsonl")
    p.add_argument("--silver-dataset", default="")
    p.add_argument("--features", default="alsdrop/data/features_manifest.jsonl")
    p.add_argument("--candidates_dir", default="alsdrop/data/candidates")
    p.add_argument("--out", default="alsdrop/models/model.pt")
    p.add_argument("--metrics", default="alsdrop/outputs/train_metrics.json")
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-neg", type=int, default=96)
    p.add_argument("--max-hard", type=int, default=96)
    p.add_argument("--rank-weight", type=float, default=0.5)
    p.add_argument("--listwise-weight", type=float, default=0.85)
    p.add_argument("--bce-weight", type=float, default=0.25)
    p.add_argument("--offset-weight", type=float, default=0.35)
    p.add_argument("--primary-weight", type=float, default=1.0)
    p.add_argument("--silver-weight", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-tracks", type=int, default=0)
    p.add_argument("--tensor-cache-capacity", type=int, default=0)
    p.add_argument("--tracks-per-step", type=int, default=8)
    p.add_argument("--model-width", type=int, default=128)
    p.add_argument("--model-dropout", type=float, default=0.15)
    p.add_argument("--long-frames", type=int, default=384)
    p.add_argument("--short-frames", type=int, default=128)
    p.add_argument("--max-candidates", type=int, default=192)
    p.add_argument("--candidate-tol-sec", type=float, default=0.100)
    p.add_argument("--no-require-candidate-hit", action="store_true")
    p.add_argument("--no-madmom", action="store_true")
    p.add_argument("--force-candidates", action="store_true")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="Predict 1.1.1 for one audio")
    p.add_argument("--audio", required=True)
    p.add_argument("--model", default="alsdrop/models/model.pt")
    p.add_argument("--out", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-madmom", action="store_true")
    p.add_argument("--bpm", type=float, default=0.0)
    p.add_argument("--review-threshold", type=float, default=0.55)
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("write-als", help="Write predicted marker to ALS template")
    p.add_argument("--template", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--pred", default="")
    p.add_argument("--pred-sec", type=float, default=None)
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--apply-to-all", action="store_true")
    p.set_defaults(func=_cmd_write_als)

    p = sub.add_parser("validate-als", help="Validate generated ALS")
    p.add_argument("--als", required=True)
    p.add_argument("--expected-sec", type=float, default=None)
    p.set_defaults(func=_cmd_validate_als)

    p = sub.add_parser("evaluate", help="Evaluate model on labeled dataset")
    p.add_argument("--dataset", default="alsdrop/data/dataset.jsonl")
    p.add_argument("--model", default="alsdrop/models/model.pt")
    p.add_argument("--out", default="alsdrop/outputs/eval_metrics.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-madmom", action="store_true")
    p.add_argument("--review-threshold", type=float, default=0.55)
    p.add_argument("--candidate-tol-sec", type=float, default=0.100)
    p.add_argument("--debug-dir", default="")
    p.add_argument("--report-html", default="")
    p.add_argument("--seen-dataset", default="")
    p.add_argument("--split-from-model", default="")
    p.add_argument("--split-name", default="test")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("tier", help="Build GOLD/SILVER/BRONZE tiered datasets")
    p.add_argument("--dataset", default="alsdrop/data/dataset.jsonl")
    p.add_argument("--out-gold", default="alsdrop/data/dataset_gold.jsonl")
    p.add_argument("--out-silver", default="alsdrop/data/dataset_silver.jsonl")
    p.add_argument("--out-bronze", default="alsdrop/data/dataset_bronze.jsonl")
    p.add_argument("--report-json", default="alsdrop/outputs/dataset_tier_report.json")
    p.add_argument("--report-html", default="alsdrop/outputs/dataset_tier_report.html")
    p.add_argument("--features-manifest", default="")
    p.add_argument("--no-drums-only", action="store_true")
    p.add_argument("--min-gold-rows", type=int, default=300)
    p.set_defaults(func=_cmd_tier)

    p = sub.add_parser("batch", help="Predict + write ALS for all files in a directory")
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--model", default="alsdrop/models/model.pt")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-madmom", action="store_true")
    p.add_argument("--bpm", type=float, default=0.0)
    p.add_argument("--review-threshold", type=float, default=0.55)
    p.add_argument("--apply-to-all", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=_cmd_batch)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
