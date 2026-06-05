from __future__ import annotations

from drop_aligner.pipeline import run_drop_candidate_pipeline


def _candidate(
    time_sec: float,
    *,
    rank: int,
    score: float,
    roles: list[str],
    micro: float,
    offset_ms: float = 0.0,
    fake: float = 0.0,
    drums: float = 0.0,
    inst: float = 0.0,
    bass: float = 0.0,
    vocal: float = 0.0,
) -> dict:
    return {
        "rank": rank,
        "handcrafted_rank": rank,
        "timestamp": time_sec,
        "confidence_score": score,
        "score": score,
        "fake_hit_penalty": fake,
        "drums_transient_score": drums,
        "inst_energy_jump_score": inst,
        "bass_low_jump_score": bass,
        "vocal_transition_score": vocal,
        "multistem_roles": roles,
        "multistem_agreement": min(1.0, len(set(roles) - {"saved"}) / 3.0),
        "microalign": {
            "ok": True,
            "microaligned_time": time_sec,
            "micro_confidence": micro,
            "snap_offset_ms": offset_ms,
            "visual_onset_knee_quality": micro,
            "visual_onset_knee_used": 1.0,
        },
    }


def test_pipeline_clusters_near_duplicate_microaligned_candidates() -> None:
    candidates = [
        _candidate(10.000, rank=1, score=0.70, roles=["drums"], micro=0.86, drums=0.80),
        _candidate(10.035, rank=2, score=0.74, roles=["vocals"], micro=0.91, vocal=0.90),
        _candidate(12.000, rank=3, score=0.68, roles=["instrumental"], micro=0.84, inst=0.76),
    ]

    result = run_drop_candidate_pipeline(candidates, cluster_radius_sec=0.085)
    ranked = result["candidates"]

    assert result["summary"]["input_count"] == 3
    assert result["summary"]["cluster_count"] == 2
    assert result["summary"]["deduped_count"] == 1
    assert len(ranked) == 2
    assert ranked[0]["drop_pipeline_cluster_size"] == 2
    assert set(ranked[0]["drop_pipeline_cluster_roles"]) == {"drums", "vocals"}
    assert ranked[0]["confidence_score"] == ranked[0]["drop_pipeline_score"]
    assert "final" in ranked[0]["drop_pipeline_score_components"]


def test_pipeline_prefers_clean_multistem_boundary_over_fake_hit() -> None:
    fake_hit = _candidate(
        20.0,
        rank=1,
        score=0.96,
        roles=["drums"],
        micro=0.72,
        offset_ms=110.0,
        fake=0.88,
        drums=0.92,
    )
    true_drop = _candidate(
        24.0,
        rank=2,
        score=0.76,
        roles=["drums", "instrumental", "vocals"],
        micro=0.97,
        offset_ms=4.0,
        fake=0.02,
        drums=0.84,
        inst=0.88,
        bass=0.80,
        vocal=0.74,
    )

    result = run_drop_candidate_pipeline([fake_hit, true_drop], cluster_radius_sec=0.085)
    ranked = result["candidates"]

    assert ranked[0]["timestamp"] == 24.0
    assert ranked[0]["rank"] == 1
    assert ranked[0]["drop_pipeline_score"] > ranked[1]["drop_pipeline_score"]
    assert ranked[0]["drop_pipeline_negative_penalty"] < ranked[1]["drop_pipeline_negative_penalty"]
