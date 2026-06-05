from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .learning import closest_candidate_to_pick


def default_candidate_json(audio_path: str) -> str:
    audio = Path(audio_path)
    return str(audio.with_name(f"{audio.stem}_drop_candidates.json"))


def default_debug_plot(audio_path: str) -> str:
    audio = Path(audio_path)
    return str(audio.with_name(f"{audio.stem}_drop_debug.png"))


def candidate_debug_payload(result, user_pick: Optional[float] = None) -> Dict[str, Any]:
    top_10 = result.top_candidate_dicts(10)
    payload: Dict[str, Any] = {
        "track": result.audio_path,
        "final_ai_pick": float(result.drop_sec),
        "coarse_ai_pick": float(result.coarse_sec),
        "bpm": float(result.bpm),
        "confidence": float(result.confidence),
        "confidence_tier": result.confidence_tier,
        "selected_by": result.selected_by,
        "feature_summary": dict(result.features_summary),
        "selected_candidate": result.selected_candidate_dict(),
        "top_10_candidates": top_10,
    }
    if user_pick is not None:
        payload["user_pick"] = float(user_pick)
        payload["closest_candidate_to_user_pick"] = closest_candidate_to_pick(top_10, float(user_pick))
    return payload


def write_candidate_debug_json(result, output_path: str, user_pick: Optional[float] = None) -> str:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(candidate_debug_payload(result, user_pick), fh, indent=2, ensure_ascii=True)
    return str(path)
