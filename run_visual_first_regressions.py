#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_FILE = "drop_aligner/tests/test_visual_first.py"


def _collect_nodeids(test_file: str) -> List[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "--collect-only", "-q"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return [line.strip() for line in proc.stdout.splitlines() if "::" in line]


def _run_nodeids(nodeids: Sequence[str], *, timeout_sec: float, verbose: bool = False) -> Dict[str, Any]:
    started = time.time()
    cmd = [sys.executable, "-m", "pytest", *nodeids, "-q", "-x"]
    if verbose:
        cmd.append("-vv")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=max(1.0, float(timeout_sec)))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        return {
            "status": "timeout",
            "returncode": 124,
            "elapsed_sec": time.time() - started,
            "stdout": stdout or "",
            "stderr": stderr or "",
        }
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "elapsed_sec": time.time() - started,
        "stdout": stdout,
        "stderr": stderr,
    }


def _isolate_timeout(
    nodeids: Sequence[str],
    *,
    timeout_sec: float,
    verbose: bool,
) -> Dict[str, Any]:
    if len(nodeids) <= 1:
        result = _run_nodeids(nodeids, timeout_sec=timeout_sec, verbose=verbose)
        return {
            **result,
            "isolated_nodeids": list(nodeids),
        }
    midpoint = max(1, len(nodeids) // 2)
    left = list(nodeids[:midpoint])
    right = list(nodeids[midpoint:])
    left_result = _run_nodeids(left, timeout_sec=timeout_sec, verbose=verbose)
    if left_result["status"] == "timeout":
        return _isolate_timeout(left, timeout_sec=timeout_sec, verbose=verbose)
    if left_result["status"] == "fail":
        return {**left_result, "isolated_nodeids": left}
    right_result = _run_nodeids(right, timeout_sec=timeout_sec, verbose=verbose)
    if right_result["status"] == "timeout":
        return _isolate_timeout(right, timeout_sec=timeout_sec, verbose=verbose)
    return {**right_result, "isolated_nodeids": right}


def run(args: argparse.Namespace) -> int:
    nodeids = _collect_nodeids(args.test_file)
    start = max(1, int(args.start or 1))
    stop = int(args.stop or len(nodeids))
    selected = nodeids[start - 1 : stop]
    if args.keyword:
        selected = [nodeid for nodeid in selected if args.keyword in nodeid]
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else REPO_ROOT / "artifacts" / "visual_first_regression_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"visual_first_regressions_{run_stamp}.json"

    results: List[Dict[str, Any]] = []
    chunk_size = max(1, int(args.chunk_size))
    exit_code = 0
    for offset in range(0, len(selected), chunk_size):
        chunk = selected[offset : offset + chunk_size]
        first_index = start + offset
        last_index = first_index + len(chunk) - 1
        print(f"[VISUAL-FIRST TEST] chunk {first_index}-{last_index} ({len(chunk)} tests)", flush=True)
        result = _run_nodeids(chunk, timeout_sec=float(args.timeout_sec), verbose=bool(args.verbose_pytest))
        result.update(
            {
                "first_index": first_index,
                "last_index": last_index,
                "nodeids": chunk,
            }
        )
        if result["status"] == "timeout" and bool(args.isolate_timeout):
            isolated = _isolate_timeout(chunk, timeout_sec=float(args.timeout_sec), verbose=bool(args.verbose_pytest))
            result["isolated_timeout"] = isolated
            if isolated.get("isolated_nodeids"):
                print("[VISUAL-FIRST TEST] isolated timeout/failure:")
                for nodeid in isolated["isolated_nodeids"]:
                    print(f"  {nodeid}")
        if result["stdout"]:
            print(result["stdout"], end="" if result["stdout"].endswith("\n") else "\n")
        if result["stderr"]:
            print(result["stderr"], end="" if result["stderr"].endswith("\n") else "\n", file=sys.stderr)
        results.append(result)
        if result["status"] != "pass":
            exit_code = int(result.get("returncode") or 1)
            break

    summary = {
        "test_file": args.test_file,
        "total_collected": len(nodeids),
        "selected_count": len(selected),
        "start": start,
        "stop": stop,
        "chunk_size": chunk_size,
        "timeout_sec": float(args.timeout_sec),
        "passed": exit_code == 0,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[VISUAL-FIRST TEST] summary={summary_path}", flush=True)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run visual-first pytest regressions in bounded chunks so slow audio cases cannot hang Codex."
    )
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--start", type=int, default=1, help="1-based first collected test index.")
    parser.add_argument("--stop", type=int, default=0, help="1-based last collected test index; 0 means all.")
    parser.add_argument("--keyword", default="", help="Substring filter applied to collected node ids.")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--isolate-timeout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose-pytest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
