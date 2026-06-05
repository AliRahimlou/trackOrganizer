from __future__ import annotations

import importlib
from pathlib import Path

import project_config


def test_project_config_derives_defaults_from_music_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRACK_ORGANIZER_MUSIC_ROOT", str(tmp_path / "MusicRoot"))

    cfg = importlib.reload(project_config)

    assert cfg.MUSIC_ROOT == tmp_path / "MusicRoot"
    assert cfg.PLAYLISTS_BY_DATE_DIR == tmp_path / "MusicRoot" / "PlaylistsByDate"
    assert cfg.STEMS_ROOT_DIR == tmp_path / "MusicRoot" / "STEMS"
    assert cfg.STEMS_INBOX_DIR == tmp_path / "MusicRoot" / "STEMS" / "toBeOrganized"
    assert cfg.DROP_BATCH_SUMMARY == tmp_path / "MusicRoot" / "STEMS" / "drop_batch_summary.csv"


def test_project_config_specific_env_overrides_parent_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRACK_ORGANIZER_MUSIC_ROOT", str(tmp_path / "MusicRoot"))
    monkeypatch.setenv("TRACK_ORGANIZER_STEMS_DIR", str(tmp_path / "CustomStems"))

    cfg = importlib.reload(project_config)

    assert cfg.STEMS_ROOT_DIR == tmp_path / "CustomStems"
    assert cfg.STEMS_INBOX_DIR == tmp_path / "CustomStems" / "toBeOrganized"
    assert cfg.DROP_BATCH_SUMMARY == tmp_path / "CustomStems" / "drop_batch_summary.csv"
