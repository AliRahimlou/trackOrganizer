from __future__ import annotations

from pathlib import Path

from project_audit import audit_project


def test_audit_project_counts_source_and_risks(tmp_path: Path) -> None:
    (tmp_path / "tool.py").write_text(
        'ROOT = "/Users/example/Music"\n'
        "try:\n"
        "    work()\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_reports").mkdir()
    (tmp_path / "eval_reports" / "run.json").write_text("{}", encoding="utf-8")

    payload = audit_project(str(tmp_path), max_findings_per_kind=20)

    assert payload["source_file_count"] == 1
    assert payload["python_file_count"] == 1
    assert payload["risk_counts"]["hardcoded_local_path"] == 1
    assert payload["risk_counts"]["broad_exception"] == 1
    assert payload["risk_counts"]["silent_pass"] == 1
    assert payload["generated_dirs_present"] == ["eval_reports"]
