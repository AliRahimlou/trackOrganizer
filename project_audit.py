from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


IGNORE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".review_previews",
    ".ruff_cache",
    ".venv",
    ".waveform_cache",
    "__pycache__",
    "alsdrop_review",
    "bin",
    "eval_reports",
    "models",
    "node_modules",
    "venv",
    "venv311",
}

IGNORE_PREFIXES = {
    "als_detector/data/",
    "als_detector/models/",
    "als_detector/outputs/",
    "alsdrop/data/",
    "alsdrop/models/",
    "alsdrop/outputs/",
}

IGNORE_FILES = {
    "drop_marker_db.json",
    "drop_self_learning_model.json",
}

SOURCE_EXTS = {
    ".command",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".ts",
}

RISK_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "hardcoded_local_path": re.compile(r"/(?:Users|Volumes)/[A-Za-z0-9_./ -]+"),
    "broad_exception": re.compile(r"\bexcept\s+Exception\b"),
    "silent_pass": re.compile(r"^\s*pass\s*(?:#.*)?$"),
    "todo_marker": re.compile(r"\b(?:TODO|FIXME|HACK|XXX)\b"),
}


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    line: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "line": int(self.line),
            "text": self.text,
        }


def _is_ignored(path: Path) -> bool:
    rel = path.as_posix()
    return (
        path.name in IGNORE_FILES
        or any(part in IGNORE_DIRS for part in path.parts)
        or any(rel.startswith(prefix) for prefix in IGNORE_PREFIXES)
    )


def iter_source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_ignored(path.relative_to(root)):
            continue
        if path.suffix.lower() in SOURCE_EXTS:
            yield path


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
    except OSError:
        return None


def scan_findings(root: Path, *, max_per_kind: int = 120) -> List[Finding]:
    counts: Dict[str, int] = {kind: 0 for kind in RISK_PATTERNS}
    findings: List[Finding] = []
    for path in iter_source_files(root):
        text = _read_text(path)
        if text is None:
            continue
        rel = str(path.relative_to(root))
        for line_no, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in RISK_PATTERNS.items():
                if counts[kind] >= int(max_per_kind):
                    continue
                if pattern.search(line):
                    if kind == "todo_marker" and rel == "project_audit.py" and "todo_marker" in line:
                        continue
                    counts[kind] += 1
                    findings.append(Finding(kind=kind, path=rel, line=line_no, text=line.strip()[:220]))
    return findings


def _file_groups(files: List[Path], root: Path) -> Dict[str, int]:
    groups: Dict[str, int] = {}
    for path in files:
        rel = path.relative_to(root)
        key = rel.parts[0] if len(rel.parts) > 1 else "."
        groups[key] = groups.get(key, 0) + 1
    return dict(sorted(groups.items(), key=lambda row: (-row[1], row[0])))


def audit_project(root: str = ".", *, max_findings_per_kind: int = 120) -> Dict[str, Any]:
    base = Path(root).resolve()
    files = list(iter_source_files(base))
    py_files = [path for path in files if path.suffix == ".py"]
    test_files = [path for path in py_files if path.name.startswith("test_") or "/tests/" in str(path.relative_to(base))]
    entrypoints = []
    for path in py_files:
        text = _read_text(path) or ""
        if 'if __name__ == "__main__"' in text:
            entrypoints.append(str(path.relative_to(base)))

    findings = scan_findings(base, max_per_kind=max_findings_per_kind)
    by_kind: Dict[str, int] = {}
    for finding in findings:
        by_kind[finding.kind] = by_kind.get(finding.kind, 0) + 1

    generated_dirs = [
        str(path.relative_to(base))
        for path in (
            base / ".waveform_cache",
            base / "eval_reports",
            base / "models",
            base / ".review_previews",
            base / "tools" / "ableton-warp-probe" / "node_modules",
        )
        if path.exists()
    ]
    return {
        "root": str(base),
        "source_file_count": int(len(files)),
        "python_file_count": int(len(py_files)),
        "test_file_count": int(len(test_files)),
        "entrypoint_count": int(len(entrypoints)),
        "entrypoints": entrypoints[:80],
        "source_groups": _file_groups(files, base),
        "risk_counts": {kind: int(by_kind.get(kind, 0)) for kind in sorted(RISK_PATTERNS)},
        "findings": [finding.to_dict() for finding in findings],
        "generated_dirs_present": generated_dirs,
        "audit_notes": [
            "Generated data/model/report directories should remain ignored or move to artifact storage.",
            "Hardcoded local paths should move behind environment variables or a project config file.",
            "Broad exception handlers are acceptable at provider boundaries but should expose status metadata.",
        ],
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Project Audit",
        "",
        f"- Source files: {payload.get('source_file_count', 0)}",
        f"- Python files: {payload.get('python_file_count', 0)}",
        f"- Test files: {payload.get('test_file_count', 0)}",
        f"- Entrypoints: {payload.get('entrypoint_count', 0)}",
        "",
        "## Risk Counts",
        "",
    ]
    risk_counts = payload.get("risk_counts") if isinstance(payload.get("risk_counts"), Mapping) else {}
    for key, value in sorted(risk_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Findings", ""])
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    for row in findings[:60]:
        if not isinstance(row, Mapping):
            continue
        lines.append(f"- {row.get('kind')} `{row.get('path')}:{row.get('line')}` {row.get('text')}")
    lines.extend(["", "## Generated Directories Present", ""])
    for item in payload.get("generated_dirs_present", []) or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Notes", ""])
    for note in payload.get("audit_notes", []) or []:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit TrackOrganizer source layout and common maintainability risks.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", dest="json_path", help="Write audit JSON")
    parser.add_argument("--markdown", dest="markdown_path", help="Write markdown summary")
    parser.add_argument("--max-findings-per-kind", type=int, default=120)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_project(args.root, max_findings_per_kind=int(args.max_findings_per_kind))
    if args.json_path:
        path = Path(args.json_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.markdown_path:
        path = Path(args.markdown_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
