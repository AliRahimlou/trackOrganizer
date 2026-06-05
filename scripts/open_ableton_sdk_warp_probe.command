#!/bin/zsh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROBE_DIR="$REPO_DIR/tools/ableton-warp-probe"
LIVE_APP="/Applications/Ableton Live 12 Beta.app"

if [ ! -d "$PROBE_DIR/node_modules" ]; then
  cd "$PROBE_DIR"
  npm install
fi

cd "$PROBE_DIR"
echo "Starting TrackOrganizer Ableton SDK warp probe..."
echo "Live app: $LIVE_APP"
echo
echo "In Live, right-click:"
echo "  - a clip slot: TrackOrganizer: Export Slot Warp Markers"
echo "  - selected slots: TrackOrganizer: Export Selected Session Warp Markers"
echo "  - CH1/CH2/CH3 track: TrackOrganizer: Export Track Warp Markers"
echo "  - any audio track: TrackOrganizer: Export Set Warp Markers"
echo
npm start -- --live "$LIVE_APP"
