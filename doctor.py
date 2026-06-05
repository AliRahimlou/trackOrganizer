#!/usr/bin/env python3
from __future__ import annotations

import gzip
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


REQUIRED_PACKAGES = [
    ("mutagen", "mutagen"),
    ("librosa", "librosa"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("lxml", "lxml"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
]


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(status, 2)


def _print_check(check: Check) -> None:
    detail = f" - {check.detail}" if check.detail else ""
    print(f"{check.status}: {check.name}{detail}")


def _check_python() -> Check:
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro} at {sys.executable}"
    if version >= (3, 10):
        return Check("PASS", "Python version", text)
    if version >= (3, 9):
        return Check("WARN", "Python version", f"{text}; Python 3.10+ recommended")
    return Check("FAIL", "Python version", f"{text}; Python 3.10+ required")


def _check_packages() -> List[Check]:
    checks: List[Check] = []
    for package_name, import_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "")
            suffix = f" {version}" if version else ""
            checks.append(Check("PASS", f"package {package_name}", f"installed{suffix}"))
        except Exception as exc:
            checks.append(Check("FAIL", f"package {package_name}", f"missing import {import_name}: {exc}"))
    return checks


def _template_candidates(root: Path) -> List[Path]:
    preferred = [
        root / "alsFiles" / "128.als",
        root / "stuff" / "CH1.als",
        root / "CH1.als",
    ]
    seen = set()
    out: List[Path] = []
    for path in preferred:
        if path.exists() and path not in seen:
            out.append(path)
            seen.add(path)
    for path in sorted((root / "alsFiles").glob("*.als")) if (root / "alsFiles").exists() else []:
        if path not in seen:
            out.append(path)
            seen.add(path)
    for path in sorted(root.glob("**/CH1.als")):
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _readable_als(path: Path) -> bool:
    from lxml import etree

    with gzip.open(path, "rb") as fh:
        data = fh.read()
    etree.fromstring(data)
    return True


def _check_template(root: Path) -> Check:
    candidates = _template_candidates(root)
    if not candidates:
        return Check("WARN", "Ableton template readable", "no .als template found in alsFiles/ or CH1.als")
    errors = []
    for path in candidates:
        try:
            _readable_als(path)
            return Check("PASS", "Ableton template readable", str(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return Check("FAIL", "Ableton template readable", "; ".join(errors[:3]))


def _check_model(root: Path) -> Check:
    path = root / "models" / "drop_ranker.pkl"
    if path.exists():
        return Check("PASS", "trained ranker model", str(path))
    return Check("WARN", "trained ranker model", "no trained model yet, using handcrafted ranking")


def _check_corrections(root: Path) -> Check:
    path = root / "drop_corrections.jsonl"
    if path.exists() and path.stat().st_size > 0:
        return Check("PASS", "correction log", str(path))
    if path.exists():
        return Check("WARN", "correction log", "drop_corrections.jsonl exists but is empty")
    return Check("WARN", "correction log", "no correction log yet, review batch results")


def _iter_recent_als_candidates(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*_DROP_ALIGNED.als"):
                if path.is_file():
                    yield path
        except PermissionError:
            continue


def _recent_generated_als(root: Path) -> Optional[Path]:
    roots = [root, Path.home() / "Desktop" / "tempSTEMs"]
    candidates = list(_iter_recent_als_candidates(roots))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _candidate_json_for_als(als_path: Path) -> Optional[Path]:
    stem = als_path.name
    suffix = "_DROP_ALIGNED.als"
    if not stem.endswith(suffix):
        return None
    base = stem[: -len(suffix)]
    candidate = als_path.with_name(f"{base}_drop_candidates.json")
    return candidate if candidate.exists() else None


def _check_recent_als(root: Path) -> Check:
    als_path = _recent_generated_als(root)
    if als_path is None:
        return Check("WARN", "ALS verification working", "no recent *_DROP_ALIGNED.als found")
    try:
        from verify_als import verify_als

        candidates_json = _candidate_json_for_als(als_path)
        report = verify_als(str(als_path), candidates_json=str(candidates_json) if candidates_json else None)
        if report.get("valid"):
            detail = str(als_path)
            if candidates_json:
                detail += f" with {candidates_json.name}"
            return Check("PASS", "ALS verification working", detail)
        errors = "; ".join(str(err) for err in report.get("errors", []))
        return Check("FAIL", "ALS verification working", f"{als_path}: {errors}")
    except Exception as exc:
        return Check("FAIL", "ALS verification working", str(exc))


def _recommended_next_action(checks: Sequence[Check]) -> str:
    failed = [check for check in checks if check.status == "FAIL"]
    if failed:
        first = failed[0]
        if first.name.startswith("package"):
            return "Install missing requirements: python3 -m pip install -r requirements.txt"
        return f"Fix failing check: {first.name}"
    names = {check.name: check for check in checks}
    if names.get("trained ranker model", Check("", "")).status == "WARN":
        if names.get("correction log", Check("", "")).status == "PASS":
            return "Train the first ranker: python3 train_ranker.py --corrections drop_corrections.jsonl"
        return "Run batch.py, then review.py to create correction data"
    if names.get("correction log", Check("", "")).status == "WARN":
        return "Review batch results to keep improving the model: python3 review.py drop_batch_summary.csv"
    if names.get("ALS verification working", Check("", "")).status == "WARN":
        return "Run a batch or single-track generation, then verify the generated ALS"
    return "System looks ready. Run batch.py on a folder, then review LOW/MEDIUM results."


def main() -> int:
    root = _repo_root()
    checks: List[Check] = []
    checks.append(_check_python())
    checks.extend(_check_packages())
    checks.append(_check_template(root))
    checks.append(_check_model(root))
    checks.append(_check_corrections(root))
    checks.append(_check_recent_als(root))

    for check in checks:
        _print_check(check)

    next_action = _recommended_next_action(checks)
    print(f"\nNEXT: {next_action}")

    worst = max((_status_rank(check.status) for check in checks), default=0)
    return 1 if worst == 2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
