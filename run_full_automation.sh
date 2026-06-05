#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project Python runtime: $PYTHON" >&2
  echo "Create it with: python3 -m venv venv && venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

export USE_DROP_FUSION_AUTOMATION="${USE_DROP_FUSION_AUTOMATION:-1}"
export DROP_FUSION_REQUIRE_SAFE="${DROP_FUSION_REQUIRE_SAFE:-1}"
export DROP_FUSION_OVERRIDE_DB_ANCHOR="${DROP_FUSION_OVERRIDE_DB_ANCHOR:-1}"
export DROP_FUSION_OVERRIDE_MANUAL_ANCHOR="${DROP_FUSION_OVERRIDE_MANUAL_ANCHOR:-0}"
export USE_SOURCE_AUDIO_MIK_CUE_TAGS="${USE_SOURCE_AUDIO_MIK_CUE_TAGS:-1}"
export USE_REKORDBOX_MIK_CUE_PRIOR="${USE_REKORDBOX_MIK_CUE_PRIOR:-1}"
export TRACK_ORGANIZER_PARALLEL="${TRACK_ORGANIZER_PARALLEL:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
export REPAIR_EXISTING_LIBRARY_ONLY="${REPAIR_EXISTING_LIBRARY_ONLY:-0}"

printf 'yes\n' | "$PYTHON" trackOrganizerAndAlsGen.py
