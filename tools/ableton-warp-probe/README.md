# TrackOrganizer Ableton Warp Probe

This extension exports Ableton Live's visible audio clip warp-marker result to JSON.
It does not reverse-engineer Live's private `.asd` analyzer. It uses the public
Extensions SDK to read `AudioClip.warpMarkers`, `warping`, `warpMode`, clip
markers, and set tempo.

## Setup

The SDK tarballs are expected in `vendor/`:

```sh
unzip -j /Users/alirahimlou/Downloads/extensions-sdk-1.0.0-beta.0.zip \
  ableton-extensions-sdk-1.0.0-beta.0.tgz \
  ableton-extensions-cli-1.0.0-beta.0.tgz \
  -d tools/ableton-warp-probe/vendor
cd tools/ableton-warp-probe
npm install
```

## Run

```sh
npm start -- --live "/Applications/Ableton Live 12 Beta.app"
```

In Live:

- Right-click an audio clip and choose `TrackOrganizer: Export Warp Markers`.
- Right-click a Session View clip slot and choose
  `TrackOrganizer: Export Slot Warp Markers`.
- Select Session View clip slots and choose
  `TrackOrganizer: Export Selected Session Warp Markers`.
- Right-click an audio track and choose `TrackOrganizer: Export Track Warp Markers`.
- Right-click an audio track and choose `TrackOrganizer: Export Set Warp Markers`
  to export every arrangement and session audio clip in the open Set.
- For a selected arrangement time range on an audio track, choose
  `TrackOrganizer: Export Selection Warp Markers`.

For a double-click launcher from the repo root, run:

```sh
scripts/open_ableton_sdk_warp_probe.command
```

The extension writes a sidecar beside the source file:

```text
track.wav.ableton_warp_markers.json
```

It also writes a backup copy under the extension storage directory when Live
provides one. The Python pipeline reads the sidecar automatically.
