from __future__ import annotations

from pathlib import Path


APP_JS = Path(__file__).resolve().parents[2] / "static" / "app.js"


def test_visual_first_gui_uses_server_boom_masks_for_clicks_and_drawing() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "function serverBoomSeries" in source
    assert "boom-bars-v17-mask-segment-render" in source
    assert "const BOOM_PLACE_EDGE_GRACE_SECONDS = 0.080;" in source
    assert "function boomFrontEdgeWindowOk" in source
    assert "server_boom_mask_not_placeable" in source
    assert "missing_server_boom_placeable_mask" in source
    assert "function serverBoomDisplayActive" in source
    assert "function serverBoomActiveRuns" in source
    assert "boom_relevant_mask" in source
    assert "relevantMask: normalizedRelevantMask" in source
    assert "return Boolean(serverSeries.relevantMask[safeIndex]);" in source
    assert "serverBoomDisplayActive(serverSeries, index, tile)" in source
    assert "const BOOM_GUI_BODY_VISIBLE_DENSITY = 0.220;" in source
    assert "const BOOM_GUI_FRONT_EDGE_TICK_SECONDS = 0.020;" in source
    assert "if (boomMode && visualFirstMode() && !serverSeries) return;" in source
    assert "if (visualFirstMode() && !serverSeries)" in source
    assert "if (serverSeries && !serverPlaceable && density < BOOM_GUI_BODY_VISIBLE_DENSITY) continue;" in source
    assert "if (visualFirstMode() && !serverBoomDisplayActive(serverSeries, i, tile)) continue;" in source
    assert "const runs = serverSeries ? serverBoomActiveRuns(serverSeries, rms.length, tile)" in source
    assert ": density" in source
    assert "if (Array.isArray(tile?.boom_placeable_mask)) return tile.boom_placeable_mask.some(Boolean);" in source
    assert "reason = \"not_boom_front_edge\"" in source
    assert "function visualDirectPlaceBlockReason" in source
    assert "return \"no_boom_front_edge\"" in source
    assert "NO BOOM EDGE" in source
    assert "Waiting for the current Boom mask before placing 1.1.1." in source


def test_visual_first_gui_masks_deep_sample_inspection_to_boom_regions() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "function serverBoomInspectionActiveAtTime" in source
    assert "const serverSeries = visualFirstMode() ? serverBoomSeries(tile) : null;" in source
    assert "if (visualFirstMode() && !serverSeries) return false;" in source
    assert "serverBoomInspectionActiveAtTime(tile, time, serverSeries, 1)" in source
    assert "segments.forEach((points) => {" in source
    assert "if (currentSegment.length >= 2) segments.push(currentSegment);" in source
    assert "if (!segments.length) return;" in source
    assert "currentSegment.push([timeToX(time), sampleY(sample, laneTop, laneHeight, tile), sample]);" in source
    assert "segments.forEach((points) => {" in source
    assert "serverBoomActiveRuns(serverSeries, Math.min(mins.length, maxs.length), tile)" in source
    assert "serverBoomActiveRuns(serverSeries, rms.length, tile)" in source


def test_visual_first_gui_does_not_draw_full_width_quiet_centerline() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "function drawBoomMaskedCenterline" in source
    assert "serverBoomActiveRuns(serverSeries, length, tile).forEach" in source
    assert "if (visualFirstMode() && !serverSeries) {\n    ctx.restore();\n    return true;\n  }\n  if (visualFirstMode() && serverSeries) {" in source
    assert "drawBoomMaskedCenterline(ctx, width, mid, serverSeries, body.length, tile, start, binSpan);" in source
    assert source.index("if (visualFirstMode() && !serverSeries)") < source.index("ctx.moveTo(0, Math.round(mid) + 0.5);")


def test_visual_first_gui_highlights_are_front_edge_locked() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "const snap = closestServerBoomMaskTime(tile, marker, \"boom_placeable_mask\");" in source
    assert "const markerOnServer = serverBoomDisplayActive(serverSeries, markerIndex, tile);" in source
    assert "actualBodyRelief || markerOnServer || !snap" in source
    assert "const serverActive = serverBoomDisplayActive(serverSeries, index, tile);" in source
    assert "let endIndex = Math.max(startIndex, snap ? Math.round(Number(snap.runEnd) || startIndex) : startIndex);" in source
    assert "firstPlaceableIndex" in source
    assert "if (serverSeries && chunk.placeableCount <= 0) return;" in source
    assert "const renderStart = serverSeries && Number.isFinite(chunk.firstPlaceableIndex)" in source
    assert "start + chunk.firstPlaceableIndex * binSpan" in source


def test_visual_first_gui_snaps_and_blocks_manual_placement_from_server_mask() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "function closestServerBoomMaskTime" in source
    assert "function boomMaskFrontEdgeSnap" in source
    assert "function resolveBoomPlacementTime" in source
    assert "async function validateVisualPlacementOnServer(markerTime, options = {})" in source
    assert "/api/validate_visual_marker" in source
    assert "const allowBlueContext = context === \"blue\";" in source
    assert "const candidateForValidation = allowBlueContext" in source
    assert ": null;" in source
    assert "context," in source
    assert "BOOM_PLACE_SNAP_MAX_SECONDS" in source
    assert "BOOM_PLACE_SNAP_VIEW_FRACTION" in source
    assert "placeValidationInFlight" in source
    assert "setUserPick(placement.time)" in source
    assert "Save blocked:" in source
    assert "Snapped from" in source
    assert "hoverPlaceTime = placement.time" in source


def test_visual_first_gui_preflights_blue_acceptance() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "Checking Boom proof for blue marker" in source
    assert "const blue = blueMarkerInfo();" in source
    assert "const blueCandidate = blue.candidate || currentItem.selected_candidate || null;" in source
    assert "const serverValidation = await validateVisualPlacementOnServer(blueTime, { context: \"blue\", candidate: blueCandidate })" in source
    assert "Blue accept blocked:" in source
    assert "Accepting validated blue marker" in source
    assert "marker_time: visualFirstMode() ? blueTime : undefined" in source
    assert "picked_candidate: visualFirstMode() ? cloneCandidateForCorrection(blueCandidate, blueTime) : undefined" in source


def test_visual_first_gui_preflights_green_save_before_correct_write() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "Checking Boom proof before saving" in source
    assert "Save blocked:" in source
    assert "not a validated Boom front edge" in source
    assert "Server Boom proof rejected the marker." in source
    assert "const candidateForLearning = visualFirstMode() ? null : pickedCandidate || closestCandidateNearTime(userPick, 0.12);" in source
    assert source.index("const serverValidation = await validateVisualPlacementOnServer(userPick)") < source.index('fetchJson("/api/correct"')


def test_visual_first_gui_busy_state_includes_marker_validation() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "function reviewUiBusy()" in source
    assert "reviewActionInFlight || saveInFlight || placeValidationInFlight" in source
    assert "const busy = reviewUiBusy();" in source
    assert "if (reviewUiBusy()) return;" in source
    assert "const enabled = Boolean(currentItem) && !reviewUiBusy() && !blockReason;" in source


def test_visual_first_gui_shortcuts_cannot_bypass_busy_state() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "function clearUserPick() {\n  if (reviewUiBusy()) return;" in source
    assert "async function refineMarker() {\n  if (!currentItem) return;\n  if (reviewUiBusy()) return;" in source
    assert "async function autoPlace() {\n  if (!currentItem) return;\n  if (reviewUiBusy()) return;" in source
    assert "async function navigate(direction) {\n  if (reviewUiBusy()) return;" in source
    assert "function placeMode() {\n  if (reviewUiBusy()) return;" in source
