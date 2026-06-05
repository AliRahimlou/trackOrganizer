#!/bin/zsh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO_DIR/venv/bin/python"
ALS_IN="/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26 copy 7.als"
ALS_OUT="/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26 copy 7_filled.als"
ALS_SYNCED="/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26 copy 7_filled_synced.als"
ALS_VISIBLE="/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26 copy 7_drop_visible.als"
ALS_MIK="/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26 copy 7_mik_firstdrops.als"
ALS_LOCAL="/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26 copy 7_local_candidates_firstdrops.als"
ALS_TRANSIENT="/Users/alirahimlou/Desktop/X1 TEMPLATE v2 Project/OG123-158 01-07-26 copy 7_ableton_transient_firstdrops.als"
REKORDBOX_XML="/Users/alirahimlou/Documents/rekordbox_mikcues_001.xml"

if [ ! -x "$PY" ]; then
  echo "Missing repo venv Python: $PY" >&2
  exit 1
fi

cd "$REPO_DIR"
echo "Filling missing 1.1.1 anchors for:"
echo "  $ALS_IN"
echo
echo "Writing marked copy:"
echo "  $ALS_OUT"
echo
"$PY" "$REPO_DIR/fill_missing_111_from_als.py" \
  --als "$ALS_IN" \
  --out "$ALS_OUT" \
  --two-pass \
  --micro-align

echo
echo "Synchronizing drums/inst/vocals anchors:"
echo "  $ALS_SYNCED"
"$PY" "$REPO_DIR/sync_triplet_111.py" \
  --als "$ALS_OUT" \
  --out "$ALS_SYNCED"

echo
echo "Repairing Live-visible 1.1.1 loop/start display:"
echo "  $ALS_VISIBLE"
"$PY" "$REPO_DIR/repair_visible_111.py" \
  --als "$ALS_SYNCED" \
  --out "$ALS_VISIBLE"

echo
echo "Refining 1.1.1 anchors to first Mixed In Key/Rekordbox drop cues:"
echo "  $ALS_MIK"
"$PY" "$REPO_DIR/refine_111_to_mik_first_drop.py" \
  --als "$ALS_VISIBLE" \
  --out "$ALS_MIK" \
  --rekordbox-xml "$REKORDBOX_XML"

echo
echo "Applying per-track local first-drop candidates:"
echo "  $ALS_LOCAL"
"$PY" "$REPO_DIR/apply_folder_drop_candidates_to_set.py" \
  --als "$ALS_MIK" \
  --out "$ALS_LOCAL"

echo
echo "Snapping 1.1.1 anchors to nearby Ableton transient markers:"
echo "  $ALS_TRANSIENT"
"$PY" "$REPO_DIR/snap_111_to_ableton_transients.py" \
  --als "$ALS_LOCAL" \
  --out "$ALS_TRANSIENT" \
  --report-csv "${ALS_TRANSIENT%.als}_transient_snap.csv"

echo
echo "Done. Opening the Ableton-transient first-drop output location in Finder."
open -R "$ALS_TRANSIENT"
