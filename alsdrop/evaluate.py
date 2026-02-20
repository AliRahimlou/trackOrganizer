#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .audio_features import extract_features
from .infer import run_predict
from .utils import as_float, iter_jsonl, write_json


def _median(vals: List[float], default: float = 0.0) -> float:
    if not vals:
        return float(default)
    s = sorted(float(v) for v in vals)
    return float(s[len(s) // 2])


def _load_targets(dataset_jsonl: str) -> List[Dict[str, object]]:
    by_audio: Dict[str, List[Dict[str, object]]] = {}
    for r in iter_jsonl(dataset_jsonl):
        ap = str(r.get("audio_path", "")).strip()
        if not ap:
            continue
        by_audio.setdefault(os.path.abspath(ap), []).append(r)

    out: List[Dict[str, object]] = []
    for ap, rows in sorted(by_audio.items()):
        tvals = [float(v) for v in [as_float(x.get("target_sec")) for x in rows] if v is not None and v > 0]
        if not tvals:
            continue
        bvals = [float(v) for v in [as_float(x.get("bpm_hint")) for x in rows] if v is not None and v > 0]
        out.append(
            {
                "audio_path": ap,
                "target_sec": _median(tvals),
                "bpm_hint": _median(bvals, default=128.0),
            }
        )
    return out


def _normalize_audio_stem(audio_path: str) -> str:
    stem = os.path.splitext(os.path.basename(audio_path))[0].strip().lower()
    if "-" in stem and "_" in stem:
        left, right = stem.split("-", 1)
        if left.startswith(("drums_", "inst_", "vocals_")) and right.strip():
            stem = right.strip()
    return stem


def _load_seen_stems(dataset_jsonl: str) -> Set[str]:
    stems: Set[str] = set()
    if not dataset_jsonl or (not os.path.isfile(dataset_jsonl)):
        return stems
    for row in iter_jsonl(dataset_jsonl):
        ap = str(row.get("audio_path", "")).strip()
        if not ap:
            continue
        stems.add(_normalize_audio_stem(ap))
    return stems


def _load_split_filter(model_path: str, split_name: str) -> Set[str]:
    if not model_path or (not os.path.isfile(model_path)):
        return set()
    try:
        import torch  # type: ignore
    except Exception:
        return set()
    try:
        ckpt = torch.load(model_path, map_location="cpu")
    except Exception:
        return set()
    splits = dict(ckpt.get("splits") or {})
    rows = splits.get(str(split_name), [])
    if not isinstance(rows, list):
        return set()
    return {os.path.abspath(str(x)) for x in rows if str(x).strip()}


def _aggregate(rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {
            "n": 0.0,
            "candidate_recall_100ms": 0.0,
            "downbeat_acc": 0.0,
            "median_abs_error_ms": 0.0,
            "p95_abs_error_ms": 0.0,
            "mean_bar_error": 0.0,
            "top3_acc": 0.0,
            "mean_confidence": 0.0,
            "acceptance_rate": 0.0,
            "accepted_downbeat_acc": 0.0,
            "manual_review_rate": 0.0,
            "safe_fallback_rate": 0.0,
        }
    err = np.asarray([float(r["error_sec"]) for r in rows], dtype=np.float32)
    bar = np.asarray([float(r["bar_error"]) for r in rows], dtype=np.float32)
    conf = np.asarray([float(r["confidence"]) for r in rows], dtype=np.float32)
    db = np.asarray([1.0 if bool(r["downbeat_match"]) else 0.0 for r in rows], dtype=np.float32)
    top3 = np.asarray([1.0 if bool(r["top3_match"]) else 0.0 for r in rows], dtype=np.float32)
    rec = np.asarray([1.0 if bool(r["candidate_hit"]) else 0.0 for r in rows], dtype=np.float32)
    accepted = np.asarray([1.0 if bool(r["accepted"]) else 0.0 for r in rows], dtype=np.float32)
    review = np.asarray([1.0 if bool(r.get("needs_manual_review", False)) else 0.0 for r in rows], dtype=np.float32)
    fallback = np.asarray([1.0 if str(r.get("selected_by", "")) == "safe_fallback" else 0.0 for r in rows], dtype=np.float32)
    if np.any(accepted > 0.5):
        accepted_db = db[accepted > 0.5]
        accepted_acc = float(np.mean(accepted_db))
    else:
        accepted_acc = 0.0

    return {
        "n": float(len(rows)),
        "candidate_recall_100ms": float(np.mean(rec)),
        "downbeat_acc": float(np.mean(db)),
        "median_abs_error_ms": float(np.median(err) * 1000.0),
        "p95_abs_error_ms": float(np.percentile(err, 95.0) * 1000.0),
        "mean_bar_error": float(np.mean(bar)),
        "top3_acc": float(np.mean(top3)),
        "mean_confidence": float(np.mean(conf)),
        "acceptance_rate": float(np.mean(accepted)),
        "accepted_downbeat_acc": float(accepted_acc),
        "manual_review_rate": float(np.mean(review)),
        "safe_fallback_rate": float(np.mean(fallback)),
    }


def _candidate_hit(target_sec: float, candidate_times: List[float], tol_sec: float) -> Tuple[bool, float, float]:
    if not candidate_times:
        return False, 0.0, 1e9
    arr = np.asarray(candidate_times, dtype=np.float32)
    idx = int(np.argmin(np.abs(arr - float(target_sec))))
    sec = float(arr[idx])
    err = abs(sec - float(target_sec))
    return bool(err <= float(tol_sec)), float(sec), float(err)


def _topk_match(target_sec: float, top: List[Dict[str, object]], tol_sec: float, k: int = 3) -> bool:
    for r in top[: max(1, int(k))]:
        try:
            t = float(r.get("sec", 0.0))
        except Exception:
            continue
        if abs(t - float(target_sec)) <= float(tol_sec):
            return True
    return False


def _write_debug_bundle(
    debug_dir: str,
    row: Dict[str, object],
    candidate_times: List[float],
) -> Optional[str]:
    ap = str(row.get("audio_path", ""))
    if not ap or (not os.path.isfile(ap)):
        return None
    try:
        feat = extract_features(audio_path=ap, sr=22050, hop_length=256, n_mels=96)
    except Exception:
        return None

    times = feat.get("frame_times", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    onset = feat.get("onset", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    low = feat.get("low_ratio", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    novelty = feat.get("novelty", np.asarray([], dtype=np.float32)).astype(np.float32, copy=False)
    if times.size == 0:
        return None

    gt = float(row.get("target_sec", 0.0))
    pred = float(row.get("predicted_sec", 0.0))
    a = max(0.0, min(gt, pred) - 20.0)
    b = min(float(times[-1]), max(gt, pred) + 20.0)
    i0 = int(np.searchsorted(times, a, side="left"))
    i1 = int(np.searchsorted(times, b, side="right"))
    i0 = max(0, min(i0, len(times)))
    i1 = max(0, min(i1, len(times)))
    if i1 <= i0:
        return None

    ts = times[i0:i1]
    on = onset[i0:i1]
    lo = low[i0:i1]
    nv = novelty[i0:i1]
    stride = max(1, int(len(ts) // 1200))
    ts = ts[::stride]
    on = on[::stride]
    lo = lo[::stride]
    nv = nv[::stride]

    bundle = {
        "audio_path": ap,
        "target_sec": gt,
        "predicted_sec": pred,
        "candidate_sec": float(row.get("candidate_sec", 0.0)),
        "confidence": float(row.get("confidence", 0.0)),
        "failure_type": str(row.get("failure_type", "unknown")),
        "window": {"start_sec": float(a), "end_sec": float(b)},
        "times": [float(x) for x in ts.tolist()],
        "onset": [float(x) for x in on.tolist()],
        "low_ratio": [float(x) for x in lo.tolist()],
        "novelty": [float(x) for x in nv.tolist()],
        "candidate_times": [float(x) for x in candidate_times],
    }

    os.makedirs(os.path.abspath(debug_dir), exist_ok=True)
    stem = os.path.splitext(os.path.basename(ap))[0]
    out = os.path.join(os.path.abspath(debug_dir), f"{stem}.debug.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=True)
        f.write("\n")
    return out


def _write_report_html(out_html: str, summary: Dict[str, object], rows: List[Dict[str, object]]) -> None:
    failure_counts = Counter(str(r.get("failure_type", "unknown")) for r in rows)
    misses = sorted(rows, key=lambda r: float(r.get("error_sec", 0.0)), reverse=True)[:50]
    lines = [
        "<html><head><meta charset='utf-8'><title>ALSDrop Eval Report</title>",
        "<style>body{font-family:Arial,sans-serif;background:#111;color:#eee;padding:20px}",
        "table{border-collapse:collapse}th,td{padding:6px 10px;border:1px solid #333}",
        ".bad{color:#ff7676}.ok{color:#7ee787}</style></head><body>",
        "<h1>ALSDrop Evaluation</h1>",
        f"<p>Tracks: <b>{int(summary.get('n', 0))}</b></p>",
        (
            "<p>"
            f"candidate_recall@100ms=<b>{float(summary.get('candidate_recall_100ms', 0.0)):.3f}</b>, "
            f"downbeat_acc=<b>{float(summary.get('downbeat_acc', 0.0)):.3f}</b>, "
            f"p95_ms=<b>{float(summary.get('p95_abs_error_ms', 0.0)):.1f}</b>, "
            f"acceptance_rate=<b>{float(summary.get('acceptance_rate', 0.0)):.3f}</b>"
            "</p>"
        ),
        "<h2>Failure Breakdown</h2>",
        "<table><tr><th>Type</th><th>Count</th></tr>",
    ]
    for k, v in sorted(failure_counts.items(), key=lambda x: (-x[1], x[0])):
        css = "bad" if k != "none" else "ok"
        lines.append(f"<tr><td class='{css}'>{html.escape(k)}</td><td>{int(v)}</td></tr>")
    lines.extend(["</table>", "<h2>Largest Errors</h2>", "<table><tr><th>Track</th><th>Error (ms)</th><th>Type</th><th>Confidence</th><th>Accepted</th></tr>"])
    for r in misses:
        track = html.escape(os.path.basename(str(r.get("audio_path", ""))))
        err_ms = float(r.get("error_sec", 0.0)) * 1000.0
        ftype = html.escape(str(r.get("failure_type", "unknown")))
        conf = float(r.get("confidence", 0.0))
        acc = bool(r.get("accepted", False))
        lines.append(
            f"<tr><td>{track}</td><td>{err_ms:.1f}</td><td>{ftype}</td>"
            f"<td>{conf:.3f}</td><td>{'yes' if acc else 'no'}</td></tr>"
        )
    lines.extend(["</table>", "</body></html>"])
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_evaluate(
    dataset_jsonl: str,
    model_path: str,
    out_json: Optional[str] = None,
    use_madmom: bool = True,
    device: str = "auto",
    review_threshold: float = 0.55,
    limit: int = 0,
    candidate_tol_sec: float = 0.100,
    debug_dir: Optional[str] = None,
    report_html: Optional[str] = None,
    seen_dataset_jsonl: Optional[str] = None,
    split_from_model_path: Optional[str] = None,
    split_name: str = "test",
) -> Dict[str, object]:
    targets = _load_targets(dataset_jsonl)
    split_filter: Set[str] = set()
    if split_from_model_path:
        split_filter = _load_split_filter(split_from_model_path, split_name=str(split_name))
        if split_filter:
            targets = [r for r in targets if os.path.abspath(str(r.get("audio_path", ""))) in split_filter]

    seen_stems = _load_seen_stems(str(seen_dataset_jsonl or ""))
    if limit and limit > 0:
        targets = targets[: int(limit)]

    rows: List[Dict[str, object]] = []
    for row in targets:
        ap = str(row["audio_path"])
        if not os.path.isfile(ap):
            continue

        pred = run_predict(
            audio_path=ap,
            model_path=model_path,
            out_json=None,
            device=device,
            use_madmom=bool(use_madmom),
            bpm_override=as_float(row.get("bpm_hint")),
            review_threshold=float(review_threshold),
            return_candidates=True,
        )

        target = float(row["target_sec"])
        bpm = float(as_float(row.get("bpm_hint"), as_float(pred.get("bpm_used"), 128.0)) or 128.0)
        beat_sec = 60.0 / max(1e-9, bpm)

        pred_sec = float(pred["predicted_sec"])
        cand_sec = float(pred.get("candidate_sec", pred_sec))
        err = abs(pred_sec - target)
        downbeat_ok = err <= max(0.060, 0.18 * beat_sec)
        bar_err = err / max(1e-9, 4.0 * beat_sec)

        cand_times = [float(x) for x in (pred.get("candidate_times") or [])]
        cand_hit, nearest_cand_sec, nearest_cand_err = _candidate_hit(target, cand_times, tol_sec=float(candidate_tol_sec))

        tops = pred.get("top_candidates") or []
        top3 = tops if isinstance(tops, list) else []
        top3_match = _topk_match(target, top3, tol_sec=max(float(candidate_tol_sec), 0.25 * beat_sec), k=3)

        cand_choice_err = abs(cand_sec - target)
        if not cand_hit:
            failure = "candidate_missing"
        elif cand_choice_err > float(candidate_tol_sec):
            failure = "ranking_wrong"
        elif err > float(candidate_tol_sec):
            failure = "refinement_wrong"
        else:
            failure = "none"

        out_row: Dict[str, object] = {
            "audio_path": ap,
            "target_sec": target,
            "predicted_sec": float(pred_sec),
            "candidate_sec": float(cand_sec),
            "nearest_candidate_sec": float(nearest_cand_sec),
            "nearest_candidate_error_sec": float(nearest_cand_err),
            "candidate_hit": bool(cand_hit),
            "error_sec": float(err),
            "bar_error": float(bar_err),
            "downbeat_match": bool(downbeat_ok),
            "top3_match": bool(top3_match),
            "confidence": float(pred.get("confidence", 0.0)),
            "score_margin": float(pred.get("score_margin", 0.0)),
            "accepted": not bool(pred.get("needs_manual_review", False)),
            "needs_manual_review": bool(pred.get("needs_manual_review", False)),
            "selected_by": str(pred.get("selected_by", "model")),
            "failure_type": str(failure),
            "is_unseen": bool(_normalize_audio_stem(ap) not in seen_stems) if seen_stems else False,
        }

        if debug_dir and failure != "none":
            dbg = _write_debug_bundle(debug_dir=debug_dir, row=out_row, candidate_times=cand_times)
            if dbg:
                out_row["debug_bundle"] = dbg

        rows.append(out_row)

    summary = _aggregate(rows)
    seen_rows = [r for r in rows if not bool(r.get("is_unseen", False))]
    unseen_rows = [r for r in rows if bool(r.get("is_unseen", False))]
    bucket_summary = {
        "all": summary,
        "seen": _aggregate(seen_rows),
        "unseen": _aggregate(unseen_rows),
        "seen_count": int(len(seen_rows)),
        "unseen_count": int(len(unseen_rows)),
    }
    failure_counts = Counter(str(r.get("failure_type", "unknown")) for r in rows)
    manual_review = [str(r.get("audio_path", "")) for r in rows if bool(r.get("needs_manual_review", False))]
    out = {
        "summary": summary,
        "buckets": bucket_summary,
        "failure_breakdown": {k: int(v) for k, v in sorted(failure_counts.items())},
        "manual_review": manual_review,
        "split_filter_size": int(len(split_filter)),
        "split_name": str(split_name),
        "split_from_model": os.path.abspath(split_from_model_path) if split_from_model_path else None,
        "seen_reference_dataset": os.path.abspath(seen_dataset_jsonl) if seen_dataset_jsonl else None,
        "tracks": rows,
    }
    if out_json:
        write_json(out_json, out)
        out["out_json"] = os.path.abspath(out_json)
    if report_html:
        _write_report_html(report_html, summary=summary, rows=rows)
        out["report_html"] = os.path.abspath(report_html)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate ALSDrop model against ALS-supervised dataset")
    ap.add_argument("--dataset", default="alsdrop/data/dataset.jsonl")
    ap.add_argument("--model", default="alsdrop/models/model.pt")
    ap.add_argument("--out", default="alsdrop/outputs/eval_metrics.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-madmom", action="store_true")
    ap.add_argument("--review-threshold", type=float, default=0.55)
    ap.add_argument("--candidate-tol-sec", type=float, default=0.100)
    ap.add_argument("--debug-dir", default="")
    ap.add_argument("--report-html", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seen-dataset", default="", help="Reference OG dataset JSONL for seen/unseen split by audio stem")
    ap.add_argument("--split-from-model", default="", help="Filter evaluation tracks to split list from checkpoint")
    ap.add_argument("--split-name", default="test", help="train|val|test split key to load from --split-from-model")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
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
    print(json.dumps(res["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
