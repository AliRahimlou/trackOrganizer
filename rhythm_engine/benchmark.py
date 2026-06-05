from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

from .cli import _read_times
from .engine import analyze_rhythm
from .types import RhythmEngineConfig


@dataclass(frozen=True)
class BenchmarkItem:
    audio: str
    reference_beats: Optional[str] = None
    reference_downbeats: Optional[str] = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio": self.audio,
            "reference_beats": self.reference_beats,
            "reference_downbeats": self.reference_downbeats,
            "metadata": dict(self.metadata or {}),
        }


def _item_from_mapping(row: Mapping[str, Any], *, base_dir: Path) -> BenchmarkItem:
    audio = str(row.get("audio") or row.get("audio_path") or row.get("path") or "").strip()
    if not audio:
        raise ValueError("Manifest row is missing audio/audio_path/path")
    beats = row.get("reference_beats") or row.get("beats") or row.get("beats_path")
    downbeats = row.get("reference_downbeats") or row.get("downbeats") or row.get("downbeats_path")

    def resolve(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path)

    audio_path = Path(audio).expanduser()
    if not audio_path.is_absolute():
        audio_path = base_dir / audio_path
    metadata = {
        str(key): value
        for key, value in row.items()
        if key not in {"audio", "audio_path", "path", "reference_beats", "beats", "beats_path", "reference_downbeats", "downbeats", "downbeats_path"}
    }
    return BenchmarkItem(
        audio=str(audio_path),
        reference_beats=resolve(beats),
        reference_downbeats=resolve(downbeats),
        metadata=metadata,
    )


def read_manifest(path: str) -> List[BenchmarkItem]:
    manifest = Path(path).expanduser()
    base_dir = manifest.parent
    if manifest.suffix.lower() == ".jsonl":
        items: List[BenchmarkItem] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL manifest rows must be objects: {manifest}")
            items.append(_item_from_mapping(payload, base_dir=base_dir))
        return items

    with open(manifest, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [_item_from_mapping(row, base_dir=base_dir) for row in reader]


def run_benchmark(
    items: Iterable[BenchmarkItem],
    *,
    config: RhythmEngineConfig,
    output_jsonl: str,
    limit: Optional[int] = None,
) -> str:
    path = Path(output_jsonl).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            if limit is not None and count >= int(limit):
                break
            result = analyze_rhythm(
                item.audio,
                config=config,
                reference_beats=_read_times(item.reference_beats),
                reference_downbeats=_read_times(item.reference_downbeats),
            ).to_dict()
            result["benchmark_item"] = item.to_dict()
            fh.write(json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n")
            count += 1
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rhythm_engine over a reference manifest and write JSONL results.")
    parser.add_argument("manifest", help="CSV or JSONL manifest with audio and reference beat/downbeat paths")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--provider", action="append", help="Provider to run. Repeat to set order.")
    parser.add_argument("--provider-weights-json", help="Provider weights JSON")
    parser.add_argument("--no-micro-refine", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = RhythmEngineConfig(
        providers=tuple(args.provider) if args.provider else RhythmEngineConfig().providers,
        provider_weights_json=args.provider_weights_json,
        micro_refine=not bool(args.no_micro_refine),
    )
    output = run_benchmark(read_manifest(args.manifest), config=cfg, output_jsonl=args.output, limit=args.limit)
    print(json.dumps({"output": output}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
