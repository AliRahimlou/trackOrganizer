let appState = null;
let currentItem = null;
let audioPlayer = null;
let stemAudioPlayers = {};
let stemMuteState = { instrumental: false, vocals: false };
let audioClipStartSec = 0;
let audioClipEndSec = 0;
let playbackFrameId = null;
let pendingPlaybackTime = null;
let playbackMarkerTime = null;
let playbackSeekTimer = null;
let correctionMode = false;
let userPick = null;
let pickedCandidate = null;
let saveInFlight = false;
let reviewActionInFlight = false;
let placeValidationInFlight = false;
let refinedPick = null;
let refinedInfo = null;
let pendingAutoPlacePick = null;
let pendingAutoPlaceCandidate = null;
let blueAnchorCandidate = null;
let waveDuration = 0;
let viewportStart = 0;
let viewportEnd = 1;
let waveTile = null;
let waveTiles = [];
let waveView = "window";
const WAVEFORM_MODE_STORAGE_KEY = "dropReviewWaveformVisualMode";
const WAVEFORM_MODE_VERSION_KEY = "dropReviewWaveformVisualModeVersion";
const WAVEFORM_MODE_VERSION = "boom-bars-v18-vector-export";
function initialWaveformVisualMode() {
  const validModes = new Set(["boom", "rms", "peaks"]);
  try {
    const storedVersion = window.localStorage?.getItem(WAVEFORM_MODE_VERSION_KEY) || "";
    const storedMode = window.localStorage?.getItem(WAVEFORM_MODE_STORAGE_KEY) || "";
    if (storedVersion === WAVEFORM_MODE_VERSION && validModes.has(storedMode)) return storedMode;
  } catch (_) {
    // Local storage can be unavailable in restricted browser contexts.
  }
  return "boom";
}
let waveformVisualMode = initialWaveformVisualMode();
let tileRequestSeq = 0;
let tileRequestTimer = null;
let tileAbortController = null;
let drawFrameId = null;
let pinchState = null;
let gestureState = null;
let waveDragState = null;
let suppressWaveClick = false;
let hoverPlaceTime = null;
let detailPanelUserChanged = false;
let syncingDetailPanel = false;
const activePointers = new Map();
const DESKTOP_WAVE_LANE_HEIGHT = 112;
const MOBILE_WAVE_LANE_HEIGHT = 122;
const VISUAL_FIRST_DESKTOP_WAVE_LANE_HEIGHT = 150;
const VISUAL_FIRST_MOBILE_WAVE_LANE_HEIGHT = 138;
const MIN_DESKTOP_WAVE_HEIGHT = 240;
const MIN_MOBILE_WAVE_HEIGHT = 260;
const VISUAL_FIRST_MIN_DESKTOP_WAVE_HEIGHT = 420;
const VISUAL_FIRST_MIN_MOBILE_WAVE_HEIGHT = 300;
const DEFAULT_WINDOW_BEFORE = 20;
const DEFAULT_WINDOW_AFTER = 30;
const MIN_VIEW_SAMPLES = 2;
const WAVEFORM_DETAIL_MULTIPLIER = 4;
const WAVEFORM_MAX_TILE_BINS = 24000;
const HYPER_RMS_TARGET = 0.78;
const HYPER_RMS_CEILING = 0.985;
const HYPER_RMS_KNEE = 0.72;
const HYPER_RMS_MIN_MAKEUP = 0.85;
const HYPER_RMS_MAX_MAKEUP = 7.5;
const HYPER_RMS_POWER = 0.82;
const HYPER_RMS_PEAK_GHOST_ALPHA = 0.06;
const HYPER_RMS_MIN_SMOOTH_SECONDS = 0.006;
const HYPER_RMS_MAX_SMOOTH_SECONDS = 0.120;
const BOOM_BODY_SMOOTH_SECONDS = 0.180;
const BOOM_BODY_TARGET_BAR_PX = 5;
const BOOM_BODY_MIN_BAR_PX = 1.2;
const BOOM_BODY_MAX_BAR_PX = 7;
const BOOM_BODY_NOISE_FLOOR = 0.180;
const BOOM_BODY_FLOOR_BLEND = 0.74;
const BOOM_BODY_CEILING_PERCENTILE = 0.965;
const BOOM_BODY_MIN_NORMALIZED = 0.380;
const BOOM_BODY_SIDE_MIN_NORMALIZED = 0.500;
const BOOM_BODY_DARK_POWER = 0.54;
const BOOM_GUI_BODY_VISIBLE_DENSITY = 0.220;
const BOOM_GUI_BODY_STRONG_DENSITY = 0.360;
const BOOM_GUI_CONTEXT_BODY_DENSITY = 0.300;
const BOOM_GUI_CONTEXT_BODY_SCALE = 0.22;
const BOOM_GUI_FRONT_EDGE_ALPHA_BOOST = 0.18;
const BOOM_GUI_FRONT_EDGE_TICK_SECONDS = 0.020;
const BOOM_PLACE_POST_SECONDS = 0.260;
const BOOM_PLACE_PRE_SECONDS = 0.110;
const BOOM_PLACE_EDGE_GRACE_SECONDS = 0.080;
const BOOM_PLACE_SNAP_MAX_SECONDS = 0.120;
const BOOM_PLACE_SNAP_VIEW_FRACTION = 0.025;
const BOOM_PLACE_MIN_POST_NORMALIZED = 0.340;
const BOOM_PLACE_MIN_PEAK_NORMALIZED = 0.430;
const BOOM_PLACE_MIN_CONTRAST = 0.070;
const RMS_INSPECTION_MIN_SAMPLE_SPACING = 0.10;
const RMS_INSPECTION_ZERO_CROSSING_SPACING = 0.65;
const RMS_INSPECTION_PEAK_ALPHA = 0.30;
const RMS_DROP_INSPECTION_RADIUS_SECONDS = 0.006;
const VISUAL_CHUNK_MIN_ALPHA = 0.10;
const VISUAL_CHUNK_MAX_ALPHA = 0.54;
const VISUAL_CHUNK_MIN_WIDTH_PX = 3;
const VISUAL_CHUNK_MAX_VIEW_SECONDS = 600;
const VISUAL_CHUNK_NOISE_GATE = 0.22;
const VISUAL_CHUNK_CANDIDATE_MARKER_VIEW_SECONDS = 18;
const VISUAL_SELECTED_DROP_MIN_SECONDS = 0.080;
const VISUAL_SELECTED_DROP_MAX_SECONDS = 1.100;
const VISUAL_SELECTED_DROP_SMOOTH_SECONDS = 0.018;
const VISUAL_SELECTED_DROP_LOW_RUN_SECONDS = 0.035;
const VISUAL_DIRECT_PLACE_VIEW_SECONDS = 6;
const TILE_REQUEST_DEBOUNCE_MS = 120;
const RESIZE_TILE_DEBOUNCE_MS = 120;
const WHEEL_ZOOM_SENSITIVITY = 0.009;
const PINCH_ZOOM_ACCELERATION = 0.72;
const MAX_AUDIO_PREVIEW_SECONDS = 120;
const MARKER_PLAY_BEFORE_SECONDS = 2;
const MARKER_PLAY_AFTER_SECONDS = 10;
const SEEK_PLAY_BEFORE_SECONDS = 5;
const SEEK_PLAY_AFTER_SECONDS = 25;
const CLICK_SEEK_AFTER_SECONDS = 30;
const PLAYBACK_SEEK_DEBOUNCE_MS = 160;
const BLUE_EXACT_MICRO_CONFIDENCE = 0.88;
const BLUE_EXACT_MAX_SNAP_MS = 30;
const STEM_STROKES = {
  drums: "#263746",
  instrumental: "#0f6fbf",
  vocals: "#7a4cff",
};
const STEM_FILLS = {
  drums: "rgba(38, 55, 70, 0.12)",
  instrumental: "rgba(15, 111, 191, 0.12)",
  vocals: "rgba(122, 76, 255, 0.12)",
};
const PLAYBACK_STEM_ROLES = ["instrumental", "vocals"];

const $ = (id) => document.getElementById(id);

function fmtTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "--";
  const s = Number(seconds);
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  return `${m}:${rem.toFixed(3).padStart(6, "0")}`;
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `${res.status} ${res.statusText}`);
  }
  return data;
}

function setStatus(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  $("statusBox").textContent = text;
  const quickStatus = $("quickStatus");
  if (quickStatus) quickStatus.textContent = text.split("\n")[0] || "";
}

function setPlaybackStatus(value) {
  const target = $("playStatus");
  if (target) target.textContent = value || "";
}

function isMobileLayout() {
  return window.matchMedia("(max-width: 900px)").matches;
}

function usableWaveTiles() {
  return waveTiles.filter((tile) => tile && tile.ok !== false);
}

function visualFirstMode() {
  return Boolean(appState?.visual_first);
}

function visualDirectPlaceReady() {
  return Boolean(visualFirstMode() && currentItem && waveDuration && viewportDuration() <= VISUAL_DIRECT_PLACE_VIEW_SECONDS + 0.001);
}

function tileMatchesViewport(tile) {
  if (!tile || tile.ok === false) return false;
  const tolerance = Math.max(0.002, viewportDuration() * 0.002);
  return Math.abs(tileStartSec(tile) - viewportStart) <= tolerance && Math.abs(tileEndSec(tile) - viewportEnd) <= tolerance;
}

function visualDirectPlaceBlockReason() {
  if (!visualDirectPlaceReady()) return "";
  const tile = primaryDrumsTile();
  if (!tile || !tileMatchesViewport(tile)) return "loading_boom_mask";
  if (!tileHasPlaceableBoomEvidence(tile)) return "no_boom_front_edge";
  return "";
}

function availableStemCount() {
  const loaded = usableWaveTiles().length;
  if (loaded) return loaded;
  const itemStems = (currentItem?.stem_waveforms || []).filter((stem) => stem.available !== false).length;
  return Math.max(1, itemStems || 1);
}

function waveCanvasHeight() {
  const laneHeight = visualFirstMode()
    ? isMobileLayout()
      ? VISUAL_FIRST_MOBILE_WAVE_LANE_HEIGHT
      : VISUAL_FIRST_DESKTOP_WAVE_LANE_HEIGHT
    : isMobileLayout()
      ? MOBILE_WAVE_LANE_HEIGHT
      : DESKTOP_WAVE_LANE_HEIGHT;
  const minHeight = visualFirstMode()
    ? isMobileLayout()
      ? VISUAL_FIRST_MIN_MOBILE_WAVE_HEIGHT
      : VISUAL_FIRST_MIN_DESKTOP_WAVE_HEIGHT
    : isMobileLayout()
      ? MIN_MOBILE_WAVE_HEIGHT
      : MIN_DESKTOP_WAVE_HEIGHT;
  return Math.max(minHeight, availableStemCount() * laneHeight);
}

function syncDetailPanelForViewport() {
  const panel = $("detailPanel");
  if (!panel || detailPanelUserChanged) return;
  const shouldOpen = !isMobileLayout();
  if (panel.open === shouldOpen) return;
  syncingDetailPanel = true;
  panel.open = shouldOpen;
  window.setTimeout(() => {
    syncingDetailPanel = false;
  }, 0);
}

function tierClass(tier) {
  return String(tier || "").toLowerCase();
}

function drumMetric(source, key) {
  if (!source) return 0;
  if (source[key] !== undefined && source[key] !== null) return Number(source[key]) || 0;
  if (source.drumprint && source.drumprint[key] !== undefined && source.drumprint[key] !== null) {
    return Number(source.drumprint[key]) || 0;
  }
  if (source.full_groove && source.full_groove[key] !== undefined && source.full_groove[key] !== null) {
    return Number(source.full_groove[key]) || 0;
  }
  return 0;
}

function candidateTime(candidate) {
  const nestedMicro = Number(candidate?.microalign?.microaligned_time);
  if (Number.isFinite(nestedMicro) && nestedMicro > 0) return nestedMicro;
  for (const key of ["microaligned_time", "timestamp", "snapped_sec", "time_sec"]) {
    const value = Number(candidate?.[key]);
    if (Number.isFinite(value) && value > 0) return value;
  }
  return null;
}

function candidateMicroInfo(candidate) {
  return candidate?.microalign && typeof candidate.microalign === "object" ? candidate.microalign : candidate || {};
}

function candidateMicroNumber(candidate, key) {
  const micro = candidateMicroInfo(candidate);
  const value = Number(micro?.[key] ?? candidate?.[key]);
  return Number.isFinite(value) ? value : null;
}

function candidateFlag(candidate, key) {
  const micro = candidateMicroInfo(candidate);
  const value = micro?.[key] ?? candidate?.[key];
  return value === true || value === 1 || value === "1" || String(value).toLowerCase() === "true";
}

function candidateStillCurrent(candidate, toleranceSeconds = 0.025) {
  if (!candidate || !Array.isArray(currentItem?.top_10_candidates)) return false;
  const rank = Number(candidate.rank);
  if (!Number.isFinite(rank)) return false;
  const time = candidateTime(candidate);
  return currentItem.top_10_candidates.some((row) => {
    if (Number(row?.rank) !== rank) return false;
    const rowTime = candidateTime(row);
    if (time === null || rowTime === null) return true;
    return Math.abs(rowTime - time) <= toleranceSeconds;
  });
}

function selectedVisualAnchorCandidate() {
  if (!visualFirstMode() || !currentItem) return null;
  const selected = currentItem.selected_candidate || closestCandidateNearTime(currentItem.ai_pick, 0.025);
  const time = candidateTime(selected);
  if (!selected || time === null) return null;
  const selectedBy = String(selected.selected_by || "");
  const reason = String(selected.reason || "").toLowerCase();
  const aiPick = optionalMarkerTime(currentItem.ai_pick);
  const isVisualFatBlock =
    selectedBy === "visual_gui_first_fat_block" ||
    selectedBy === "visual_structure_section_guard" ||
    (selectedBy === "visual_gui_only" && reason.includes("first real fat waveform block"));
  if (!isVisualFatBlock) return null;
  if (aiPick !== null && Math.abs(time - aiPick) > 0.025) return null;
  return selected;
}

function candidateSnapOffsetMs(candidate, exactTime) {
  const explicit = candidateMicroNumber(candidate, "snap_offset_ms");
  if (explicit !== null) return explicit;
  const input = candidateMicroNumber(candidate, "input_candidate_time") || candidateMicroNumber(candidate, "coarse_timestamp");
  if (input !== null && exactTime !== null) return (Number(exactTime) - input) * 1000;
  return 0;
}

function blueExactCandidateTime(candidate, grid) {
  const exact = candidateTime(candidate);
  if (exact === null) return null;
  if (!grid?.time) return exact;

  const micro = candidateMicroInfo(candidate);
  const reason = String(micro?.reason || candidate?.microalign_reason || candidate?.reason || "");
  const microConfidence = candidateMicroNumber(candidate, "micro_confidence") || 0;
  const attackCleanliness = candidateMicroNumber(candidate, "attack_cleanliness") || 0;
  const zeroQuality = candidateMicroNumber(candidate, "zero_crossing_quality") || 0;
  const snapMs = Math.abs(candidateSnapOffsetMs(candidate, exact));
  const kneeTime = candidateMicroNumber(candidate, "visual_onset_knee_time");
  const kneeUsed = Boolean(Number(micro?.visual_onset_knee_used ?? candidate?.visual_onset_knee_used ?? 0));
  const exactIsKnee = kneeTime !== null && Math.abs(kneeTime - exact) <= 0.002;
  const tailBypass =
    /tail bypass|impact body|clock[_ -]?lock/i.test(reason) ||
    Boolean(micro?.clock_locked) ||
    (Number(micro?.impact_body_used || 0) > 0 && Math.abs(Number(micro?.impact_body_offset_ms || 0)) > 2);
  const reliableKnee =
    (kneeUsed || exactIsKnee) &&
    microConfidence >= BLUE_EXACT_MICRO_CONFIDENCE &&
    attackCleanliness >= 0.85 &&
    zeroQuality >= 0.8 &&
    snapMs <= BLUE_EXACT_MAX_SNAP_MS;
  const cleanKnee =
    /sample-level|centerline|knee|quiet pre-attack boundary/i.test(reason) &&
    microConfidence >= BLUE_EXACT_MICRO_CONFIDENCE &&
    attackCleanliness >= 0.9 &&
    zeroQuality >= 0.85 &&
    snapMs <= BLUE_EXACT_MAX_SNAP_MS;

  if (tailBypass && !reliableKnee) return null;
  if (cleanKnee || reliableKnee) return exact;
  return null;
}

function barPriorForCandidate(candidate) {
  return candidate?.musical_bar_prior || candidate?.bar_prior || null;
}

function bpmClockForCandidate(candidate) {
  return candidate?.bpm_clock || candidate?.clock || null;
}

function bpmOneClockForTime(time) {
  const bpm = Number(currentItem?.bpm || 0);
  const anchor = optionalMarkerTime(time);
  if (!Number.isFinite(bpm) || bpm <= 0 || anchor === null) return null;
  const beatSeconds = 60 / bpm;
  const barSeconds = beatSeconds * 4;
  if (!Number.isFinite(beatSeconds) || beatSeconds <= 0 || !Number.isFinite(barSeconds) || barSeconds <= 0) return null;
  const zero = barZeroSeconds();
  const index = Math.max(0, Math.round((anchor - zero) / barSeconds));
  const gridTime = zero + index * barSeconds;
  const oneDistanceSec = Math.abs(anchor - gridTime);
  return {
    bpm,
    beat_sec: beatSeconds,
    bar_sec: barSeconds,
    nearest_one_bar: index + 1,
    nearest_one_time: gridTime,
    one_distance_sec: oneDistanceSec,
    one_distance_ms: oneDistanceSec * 1000,
    one_distance_beats: oneDistanceSec / beatSeconds,
    on_one_score: Math.max(0, 1 - (oneDistanceSec / beatSeconds / 0.5)),
  };
}

function candidateBpmClock(candidate) {
  return bpmClockForCandidate(candidate) || bpmOneClockForTime(candidateTime(candidate));
}

function isHistoricalCandidate(candidate) {
  const selectedBy = String(candidate?.selected_by || "");
  const source = String(candidate?.source || "");
  const reason = String(candidate?.reason || "");
  return (
    selectedBy === "historical_human_marker" ||
    selectedBy === "historical_review_memory" ||
    source === "historical_human_marker" ||
    source === "historical_review_memory" ||
    reason.startsWith("historical ")
  );
}

function structureMap() {
  return currentItem?.structure_map || null;
}

function structureBeatgrid() {
  return currentItem?.feature_map?.beatgrid || currentItem?.beatgrid || structureMap()?.beatgrid || null;
}

function dropPointContract() {
  return currentItem?.drop_points && typeof currentItem.drop_points === "object" ? currentItem.drop_points : null;
}

function dropPointTime(name) {
  return optionalMarkerTime(dropPointContract()?.[name]?.time_sec);
}

function acceptedAlsImpactTime() {
  const points = dropPointContract();
  if (!points || points.status !== "accepted") return null;
  return dropPointTime("als_impact");
}

function alternateAttackPoints() {
  const rows = dropPointContract()?.alternate_attacks;
  return Array.isArray(rows) ? rows.filter((row) => optionalMarkerTime(row?.time_sec) !== null) : [];
}

function barZeroSeconds() {
  const zero = Number(structureBeatgrid()?.bar_zero_sec);
  return Number.isFinite(zero) && zero >= 0 ? zero : 0;
}

function structureCandidate(role) {
  const selected = currentItem?.selected_candidate || null;
  if (
    selected &&
    (selected.structure_role === role ||
      selected.section_label === role ||
      (role === "first_drop" && String(selected.selected_by || "").startsWith("structure_map")))
  ) {
    return selected;
  }
  const map = structureMap();
  if (!map) return null;
  if (role === "first_drop") return map.first_drop || null;
  if (role === "second_drop") return map.second_drop || null;
  return null;
}

function formatBarPrior(prior) {
  if (!prior) return "";
  const bars = Number(prior.nearest_musical_bar ?? prior.nearest_even_bar ?? prior.nearest_bar ?? prior.bar);
  const beats = Number(prior.distance_beats ?? prior.bar_distance_beats);
  if (!Number.isFinite(bars) || bars <= 0 || !Number.isFinite(beats)) return "";
  if (beats > 1.25) return "";
  const beatText = beats <= 0.05 ? "on ONE" : `${beats.toFixed(2)}b off`;
  return `clock b${Math.round(bars)} ${beatText}`;
}

function formatBpmClock(clock) {
  if (!clock) return "";
  const bpm = Number(clock.bpm);
  const oneMs = Number(clock.one_distance_ms);
  const beatMs = Number(clock.distance_ms);
  const oneScore = Number(clock.on_one_score || 0);
  const bar = Number(clock.nearest_one_bar || clock.bar_number);
  if (!Number.isFinite(bpm) || !Number.isFinite(beatMs)) return "";
  if (Number.isFinite(oneMs) && oneScore >= 0.90) {
    const offset = Math.abs(oneMs) <= 1 ? "on ONE" : `ONE ${oneMs.toFixed(1)}ms`;
    return `${Math.round(bpm)} BPM ${offset}${Number.isFinite(bar) ? ` b${Math.round(bar)}` : ""}`;
  }
  return `${Math.round(bpm)} BPM beat ${beatMs <= 1 ? "on" : `${beatMs.toFixed(1)}ms`}`;
}

function formatStructureMarker(candidate) {
  if (!candidate) return "--";
  const clock =
    Number(candidate.structure_clock_bar || 0) ||
    Number(candidate.structure_components?.clock_bar || 0) ||
    Number(bpmClockForCandidate(candidate)?.nearest_one_bar || 0);
  const grid = Number(candidate.structure_bar || 0);
  const clockText = clock > 0 ? `clock b${clock}` : "";
  const gridText = grid > 0 ? `grid ${grid}` : "";
  const prefix = [clockText, gridText].filter(Boolean).join(" | ");
  return `${prefix || "bar --"} ${fmtTime(candidateTime(candidate))}`;
}

function primaryAutoGate(gates, preferred = "conservative") {
  if (!gates) return null;
  return gates[preferred] || gates.normal || gates.conservative || gates.aggressive || null;
}

function gateBadgeState(gate) {
  if (!gate) return { label: "--", cls: "review" };
  if (gate.auto_accept) return { label: "SAFE AUTO", cls: "safe" };
  const flags = Array.isArray(gate.risk_flags) ? gate.risk_flags : [];
  const tier = String(gate.confidence_tier || currentItem?.confidence_tier || "").toUpperCase();
  const hardNo = tier === "LOW" || flags.length >= 3 || flags.some((flag) => /fake-hit|full-groove|immediate groove|strongly disagree|snap offset/i.test(String(flag)));
  return hardNo ? { label: "DO NOT AUTO", cls: "no" } : { label: "REVIEW RECOMMENDED", cls: "review" };
}

function renderAutoAcceptGate(gate) {
  const badge = $("autoAcceptBadge");
  if (!badge) return;
  const state = gateBadgeState(gate);
  badge.textContent = state.label;
  badge.className = `autoBadge ${state.cls}`;
  $("autoAcceptReason").textContent = gate?.reason || "--";
  const flags = Array.isArray(gate?.risk_flags) ? gate.risk_flags : [];
  $("autoAcceptRisks").textContent = flags.length ? flags.join("; ") : "--";
}

function setCorrectionMode(enabled) {
  correctionMode = Boolean(enabled);
  $("waveWrapper")?.classList.toggle("placing", correctionMode);
  $("waveWrapper")?.classList.toggle("directPlace", visualDirectPlaceReady() && correctionMode);
}

function sampleRate() {
  const primary = waveTile || usableWaveTiles()[0];
  return Number(primary?.sample_rate || currentItem?.sample_rate || 44100);
}

function minViewSeconds() {
  return MIN_VIEW_SAMPLES / Math.max(sampleRate(), 1);
}

function snapTimeToSample(value) {
  const sr = sampleRate();
  return Math.round(Number(value) * sr) / sr;
}

function clampOriginalTime(value) {
  const duration = Math.max(waveDuration || Number(currentItem?.full_duration_sec || 0), 0);
  const clamped = Math.min(Math.max(0, Number(value)), duration || Number.POSITIVE_INFINITY);
  return snapTimeToSample(clamped);
}

function selectedMicroInfo() {
  if (refinedInfo) return refinedInfo;
  const micro = currentItem?.selected_candidate?.microalign;
  if (micro) return micro;
  const features = currentItem?.feature_summary || {};
  const out = {};
  for (const key of ["microaligned_time", "attack_start_time", "zero_crossing_time", "visual_onset_knee_time", "ableton_asd_time"]) {
    const value = features[`chosen_${key}`];
    if (value !== undefined && value !== null && Number(value) > 0) out[key] = Number(value);
  }
  return out;
}

function selectedMicroTime() {
  if (refinedPick !== null) return optionalMarkerTime(refinedPick);
  const micro = selectedMicroInfo();
  return optionalMarkerTime(micro?.microaligned_time);
}

function kneeMarkerTime() {
  const micro = selectedMicroInfo();
  return optionalMarkerTime(micro?.visual_onset_knee_time) || selectedMicroTime();
}

function optionalMarkerTime(value) {
  const time = Number(value);
  return Number.isFinite(time) && time > 0 ? time : null;
}

function nearestBpmOne(time) {
  const bpm = Number(currentItem?.bpm || 0);
  if (!Number.isFinite(bpm) || bpm <= 0) return null;
  const barSeconds = 4 * 60 / bpm;
  if (!Number.isFinite(barSeconds) || barSeconds <= 0) return null;
  const anchor = optionalMarkerTime(time);
  if (anchor === null) return null;
  const zero = barZeroSeconds();
  const index = Math.max(0, Math.round((anchor - zero) / barSeconds));
  const gridTime = zero + index * barSeconds;
  return {
    time: optionalMarkerTime(gridTime),
    barNumber: index + 1,
  };
}

function visualBoundaryScore(candidate, selectedTime) {
  const time = candidateTime(candidate);
  if (time === null) return null;
  const distance = selectedTime === null ? 0 : Math.abs(time - selectedTime);
  const score = Number(candidate?.confidence_score ?? candidate?.score ?? 0);
  const micro = candidateMicroNumber(candidate, "micro_confidence") || 0;
  const attack = candidateMicroNumber(candidate, "attack_cleanliness") || 0;
  const zero = candidateMicroNumber(candidate, "zero_crossing_quality") || 0;
  const snapMs = Math.abs(candidateSnapOffsetMs(candidate, time));
  const roles = Array.isArray(candidate?.multistem_roles) ? candidate.multistem_roles.map(String) : [];
  const sources = Array.isArray(candidate?.multistem_source_names) ? candidate.multistem_source_names.map(String) : [];
  const sourceText = [...roles, ...sources, candidate?.reason || "", candidateMicroInfo(candidate)?.reason || ""].join(" ").toLowerCase();
  const hasSaved = sourceText.includes("saved");
  const hasTransition = /bar_transition|low_jump|drop|reentry/.test(sourceText);
  const hasDrums = sourceText.includes("drums");
  const sampleBoundary = /sample-level|knee|centerline/.test(sourceText);
  const tailBypass = /tail bypass|impact body/.test(sourceText);
  const strongBoundary = micro >= 0.92 || attack >= 0.90 || (hasSaved && hasTransition);
  if (!strongBoundary || score < 0.66 || snapMs > 95 || distance > 2.25) return null;
  const value =
    score +
    0.45 * micro +
    0.25 * attack +
    0.10 * zero +
    (hasSaved ? 0.12 : 0) +
    (hasTransition ? 0.12 : 0) +
    (hasDrums ? 0.08 : 0) +
    (sampleBoundary ? 0.08 : 0) -
    (tailBypass ? 0.24 : 0) -
    Math.min(0.18, snapMs / 1600) -
    Math.min(0.16, distance / 25);
  return { candidate, time, value, distance, micro, attack, sampleBoundary, tailBypass };
}

function selectedVisualEdgeIsLocked(selected, selectedTime) {
  if (!visualFirstMode() || !selected || selectedTime === null) return false;
  const selectedBy = String(selected.selected_by || "");
  const reason = String(selected.reason || "").toLowerCase();
  const micro = candidateMicroNumber(selected, "micro_confidence") || 0;
  const snapMs = Math.abs(candidateSnapOffsetMs(selected, selectedTime));
  const attack = candidateMicroNumber(selected, "attack_cleanliness") || 0;
  if (selectedBy === "visual_gui_first_fat_block" && candidateFlag(selected, "visual_body_onset_used")) return true;
  return (
    selectedBy === "visual_gui_first_fat_block" &&
    reason.includes("zoomed from visual block edge") &&
    micro >= 0.95 &&
    snapMs <= 8 &&
    attack >= 0.85
  );
}

function visualStrongBoundaryCandidate(selected, selectedTime) {
  const candidates = Array.isArray(currentItem?.top_10_candidates) ? currentItem.top_10_candidates : [];
  if (!candidates.length || selectedTime === null) return null;
  const selectedRank = selected?.rank;
  const selectedIsHistorical = isHistoricalCandidate(selected);
  const selectedEdgeLocked = selectedVisualEdgeIsLocked(selected, selectedTime);
  const selectedScore = visualBoundaryScore(selected, selectedTime)?.value ?? 0;
  const best = candidates
    .map((candidate) => visualBoundaryScore(candidate, selectedTime))
    .filter((row) => {
      if (!row || row.candidate?.rank === selectedRank) return false;
      if (selectedEdgeLocked && row.time > selectedTime && row.distance <= 0.080) return false;
      if (!selectedIsHistorical) return true;
      return row.sampleBoundary && !row.tailBypass && row.micro >= 0.9 && row.attack >= 0.85;
    })
    .sort((a, b) => {
      const valueDelta = b.value - a.value;
      if (Math.abs(valueDelta) > 0.03) return valueDelta;
      return a.distance - b.distance;
    })[0];
  const requiredGain = selectedIsHistorical ? 0.35 : 0.10;
  if (!best || best.value < selectedScore + requiredGain) return null;
  return best.candidate;
}

function phraseStrengthForBar(bar) {
  const value = Math.round(Number(bar || 0));
  if (!Number.isFinite(value) || value <= 0) return 0;
  if ((value - 1) % 32 === 0) return 1.0;
  if ((value - 1) % 24 === 0) return 0.96;
  if ((value - 1) % 16 === 0) return 0.94;
  if ((value - 1) % 8 === 0) return 0.86;
  if ((value - 1) % 12 === 0) return 0.80;
  if ((value - 1) % 4 === 0) return 0.66;
  if ((value - 1) % 2 === 0) return 0.48;
  return 0.18;
}

function candidateEvidenceValue(candidate) {
  if (!candidate) return 0;
  const score = Number(candidate?.confidence_score ?? candidate?.score ?? 0) || 0;
  const micro = candidateMicroNumber(candidate, "micro_confidence") || 0;
  const attack = candidateMicroNumber(candidate, "attack_cleanliness") || 0;
  const zero = candidateMicroNumber(candidate, "zero_crossing_quality") || 0;
  const snapMs = Math.abs(candidateSnapOffsetMs(candidate, candidateTime(candidate)));
  const fake = Number(candidate?.fake_hit_penalty || 0) || 0;
  return (
    0.28 * Math.max(0, Math.min(1, micro)) +
    0.20 * Math.max(0, Math.min(1, score)) +
    0.08 * Math.max(0, Math.min(1, attack)) +
    0.04 * Math.max(0, Math.min(1, zero)) -
    0.08 * Math.max(0, Math.min(1, fake)) -
    0.05 * Math.max(0, Math.min(1, snapMs / 220))
  );
}

function clockPhraseStrength(clock, bar, offBeats) {
  const explicit = Number(clock?.phrase_strength);
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  if (Number.isFinite(offBeats) && offBeats > 0.16) return 0;
  return phraseStrengthForBar(bar);
}

function visualPrimaryPhraseCandidate(selected, selectedTime) {
  if (!visualFirstMode() || !currentItem || selectedTime === null || isHistoricalCandidate(selected)) return null;
  const candidates = Array.isArray(currentItem.top_10_candidates) ? currentItem.top_10_candidates : [];
  if (!candidates.length) return null;
  const selectedClock = selected ? candidateBpmClock(selected) : bpmOneClockForTime(selectedTime);
  const selectedBar = Number(selectedClock?.nearest_one_bar || selectedClock?.bar_number || selectedClock?.phrase_bar || 0);
  const selectedOffBeats = Number(selectedClock?.one_distance_beats ?? selectedClock?.distance_beats);
  const selectedStrength = clockPhraseStrength(selectedClock, selectedBar, selectedOffBeats);
  const selectedIsPrimary =
    selectedBar >= 25 &&
    selectedBar <= 49 &&
    (selectedBar - 1) % 32 === 0 &&
    Number.isFinite(selectedOffBeats) &&
    selectedOffBeats <= 0.16 &&
    selectedStrength >= 0.94;
  if (selectedIsPrimary) return null;
  const selectedSecondaryPhrase =
    selectedBar >= 25 &&
    selectedBar <= 49 &&
    Number.isFinite(selectedOffBeats) &&
    selectedOffBeats <= 0.16 &&
    selectedStrength >= 0.86;
  const selectedScore = Number(selected?.confidence_score ?? selected?.score ?? 0) || 0;
  const selectedEvidence = candidateEvidenceValue(selected);
  const rows = candidates
    .map((candidate) => {
      const time = candidateTime(candidate);
      const clock = candidateBpmClock(candidate);
      const bar = Number(clock?.nearest_one_bar || clock?.bar_number || clock?.phrase_bar || 0);
      const offBeats = Number(clock?.one_distance_beats ?? clock?.distance_beats);
      const strength = clockPhraseStrength(clock, bar, offBeats);
      const score = Number(candidate?.confidence_score ?? candidate?.score ?? 0) || 0;
      const micro = candidateMicroNumber(candidate, "micro_confidence") || 0;
      const evidence = candidateEvidenceValue(candidate);
      return { candidate, time, bar, offBeats, strength, score, micro, evidence };
    })
    .filter((row) => {
      if (row.time === null || row.time <= selectedTime + 0.5) return false;
      if (!Number.isFinite(row.bar) || row.bar < 25 || row.bar > 49 || (row.bar - 1) % 32 !== 0) return false;
      if (!Number.isFinite(row.offBeats) || row.offBeats > 0.16 || row.strength < 0.94) return false;
      if (!selectedSecondaryPhrase && selectedBar > 21 && selectedStrength >= 0.94) return false;
      if (row.micro < 0.90) return false;
      if (row.score < Math.max(0.68, selectedScore - 0.12)) return false;
      if (row.evidence < selectedEvidence - 0.14) return false;
      return true;
    })
    .sort((a, b) => {
      const barDelta = a.bar - b.bar;
      if (barDelta) return barDelta;
      const strengthDelta = b.strength - a.strength;
      if (Math.abs(strengthDelta) > 0.01) return strengthDelta;
      const microDelta = b.micro - a.micro;
      if (Math.abs(microDelta) > 0.02) return microDelta;
      return (a.candidate.rank || 9999) - (b.candidate.rank || 9999);
    });
  return rows[0]?.candidate || null;
}

function visualPreferredAnchorCandidate() {
  if (!visualFirstMode() || !currentItem) return null;
  if (selectedVisualAnchorCandidate()) return null;
  const candidates = Array.isArray(currentItem.top_10_candidates) ? currentItem.top_10_candidates : [];
  if (!candidates.length) return null;
  const selected = currentItem.selected_candidate || closestCandidateNearTime(currentItem.ai_pick) || null;
  const selectedTime = candidateTime(selected) || optionalMarkerTime(currentItem.ai_pick);
  const selectedEdgeLocked = selectedVisualEdgeIsLocked(selected, selectedTime);
  const primaryPhrase = visualPrimaryPhraseCandidate(selected, selectedTime);
  if (primaryPhrase) return primaryPhrase;
  const visualBoundary = visualStrongBoundaryCandidate(selected, selectedTime);
  if (visualBoundary) return visualBoundary;
  const selectedClock = selected ? candidateBpmClock(selected) : bpmOneClockForTime(selectedTime);
  const selectedBar = Number(selectedClock?.nearest_one_bar || 0);
  const selectedOffBeats = Number(selectedClock?.one_distance_beats);
  const beatSeconds = Number(selectedClock?.beat_sec || (Number(currentItem.bpm || 0) > 0 ? 60 / Number(currentItem.bpm) : 0));
  if (
    selectedTime === null ||
    !Number.isFinite(selectedBar) ||
    selectedBar <= 0 ||
    !Number.isFinite(selectedOffBeats) ||
    selectedOffBeats <= 0.16 ||
    !Number.isFinite(beatSeconds) ||
    beatSeconds <= 0
  ) {
    return null;
  }
  const maxJump = Math.max(0.35, Math.min(0.72, 1.15 * beatSeconds));
  const selectedScore = Number(selected?.confidence_score ?? selected?.score ?? 0);
  const rescue = candidates
    .map((candidate) => {
      const time = candidateTime(candidate);
      const clock = candidateBpmClock(candidate);
      const bar = Number(clock?.nearest_one_bar || 0);
      const offBeats = Number(clock?.one_distance_beats);
      const distance = time === null ? Infinity : Math.abs(time - selectedTime);
      const score = Number(candidate.confidence_score ?? candidate.score ?? 0);
      const micro = candidateMicroNumber(candidate, "micro_confidence") || 0;
      return { candidate, time, clock, bar, offBeats, distance, score, micro };
    })
    .filter((row) => {
      if (row.time === null || row.distance <= 0.010 || row.distance > maxJump) return false;
      if (selectedEdgeLocked && row.time > selectedTime && row.distance <= 0.080) return false;
      if (!Number.isFinite(row.bar) || row.bar !== selectedBar) return false;
      if (!Number.isFinite(row.offBeats) || row.offBeats > 0.16) return false;
      if (row.score < selectedScore - 0.18 && row.micro < 0.8) return false;
      return true;
    })
    .sort((a, b) => {
      const scoreDelta = b.score - a.score;
      if (Math.abs(scoreDelta) > 0.02) return scoreDelta;
      const offDelta = a.offBeats - b.offBeats;
      if (Math.abs(offDelta) > 0.01) return offDelta;
      return (a.candidate.rank || 9999) - (b.candidate.rank || 9999);
    })[0];
  return rescue?.candidate || null;
}

function defaultBlueAnchorCandidate() {
  const selectedVisual = selectedVisualAnchorCandidate();
  if (selectedVisual) return selectedVisual;
  const liveBlueAnchor = candidateStillCurrent(blueAnchorCandidate) ? blueAnchorCandidate : null;
  const livePendingAutoPlace = candidateStillCurrent(pendingAutoPlaceCandidate) ? pendingAutoPlaceCandidate : null;
  return liveBlueAnchor || livePendingAutoPlace || visualPreferredAnchorCandidate() || currentItem?.selected_candidate || null;
}

function closestCandidateNearTime(time, maxDistance = 0.18) {
  const target = optionalMarkerTime(time);
  const candidates = Array.isArray(currentItem?.top_10_candidates) ? currentItem.top_10_candidates : [];
  if (target === null || !candidates.length) return null;
  let best = null;
  let bestDistance = Infinity;
  candidates.forEach((candidate) => {
    const marker = candidateTime(candidate);
    if (marker === null) return;
    const distance = Math.abs(marker - target);
    if (distance < bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  });
  return best && bestDistance <= maxDistance ? best : null;
}

function blueMarkerInfo() {
  const manual = visualFirstMode() || pickedCandidate || userPick === null ? null : optionalMarkerTime(userPick);
  if (manual !== null) {
    return {
      time: manual,
      label: "placed",
      barNumber: nearestBpmOne(manual)?.barNumber || null,
      candidate: null,
      source: "manual",
    };
  }
  if (visualFirstMode()) {
    const points = dropPointContract();
    const impact = acceptedAlsImpactTime();
    const time =
      optionalMarkerTime(pendingAutoPlacePick) ||
      impact ||
      (points ? null : optionalMarkerTime(currentItem?.ai_pick) || candidateTime(currentItem?.selected_candidate) || selectedMicroTime());
    if (time === null) return { time: null, label: "", barNumber: null };
    const candidate = closestCandidateNearTime(time, 0.120) || currentItem?.selected_candidate || null;
    return {
      time,
      label: impact !== null && Math.abs(time - impact) <= 1 / Math.max(sampleRate(), 1) ? "ALS" : "",
      barNumber: nearestBpmOne(time)?.barNumber || null,
      candidate,
      source: impact !== null && Math.abs(time - impact) <= 1 / Math.max(sampleRate(), 1) ? "als_impact" : "candidate",
    };
  }
  const candidate = pickedCandidate || defaultBlueAnchorCandidate();
  const candidateAnchor = candidateTime(candidate);
  const anchor = candidateAnchor || selectedMicroTime() || optionalMarkerTime(currentItem?.ai_pick);
  const grid = nearestBpmOne(anchor);
  const exact = blueExactCandidateTime(candidate, grid);
  const time = optionalMarkerTime(exact) || grid?.time || optionalMarkerTime(anchor);
  if (!time) return { time: null, label: "", barNumber: null };
  const closestCandidate = closestCandidateNearTime(time);
  const rawRank = pickedCandidate?.picked_candidate_rank ?? closestCandidate?.rank;
  const label = visualFirstMode() ? "" : rawRank ? `#${rawRank}` : "AI";
  return {
    time,
    label,
    barNumber: grid?.barNumber || null,
    candidate: closestCandidate || candidate || null,
    source: exact ? "candidate" : grid?.time ? "grid" : "anchor",
  };
}

function gridMarkerTime() {
  return blueMarkerInfo().time;
}

function visualCandidateZoomRadius() {
  const span = viewportDuration();
  if (span > 90) return 8;
  if (span > 24) return 4;
  if (span > 8) return 1;
  return Math.max(0.12, Math.min(1, span / 4));
}

function markerTimes() {
  const micro = selectedMicroInfo();
  const contract = dropPointContract();
  return {
    ai: optionalMarkerTime(pendingAutoPlacePick) || optionalMarkerTime(currentItem?.ai_pick),
    micro: selectedMicroTime(),
    attack: acceptedAlsImpactTime() || optionalMarkerTime(micro?.attack_start_time),
    zero: dropPointTime("zero_crossing") || optionalMarkerTime(micro?.zero_crossing_time),
    knee: kneeMarkerTime(),
    asd: optionalMarkerTime(micro?.ableton_asd_time),
    grid: gridMarkerTime(),
    manual: userPick === null ? null : optionalMarkerTime(userPick),
    musicalOne: dropPointTime("musical_one"),
    visual: dropPointTime("visual_marker"),
    crest: dropPointTime("body_crest"),
    alternates: Array.isArray(contract?.alternate_attacks)
      ? contract.alternate_attacks.map((row) => optionalMarkerTime(row?.time_sec)).filter((time) => time !== null)
      : [],
  };
}

function markerTime(kind) {
  return markerTimes()[kind] || null;
}

function reviewUiBusy() {
  return Boolean(reviewActionInFlight || saveInFlight || placeValidationInFlight);
}

function updateMarkerAcceptButton(id, label, time) {
  const button = $(id);
  if (!button) return;
  const busy = reviewUiBusy();
  const enabled = time !== null && !busy;
  button.disabled = !enabled;
  button.classList.toggle("disabled", !enabled);
  button.textContent = busy
    ? "WORKING..."
    : enabled
      ? `ACCEPT ${label} ${fmtTime(time)}`
      : `NO ${label} MARKER`;
}

function setReviewActionInFlight(value) {
  reviewActionInFlight = Boolean(value);
  const skipButton = $("skipBtn");
  if (skipButton) {
    const busy = reviewUiBusy();
    skipButton.disabled = busy;
    skipButton.classList.toggle("disabled", busy);
    skipButton.textContent = busy ? "WORKING..." : "SKIP";
  }
  updateSaveButton();
}

function updateAiPickDisplay() {
  const target = $("aiPick");
  if (!target) return;
  const time = markerTimes().ai;
  target.textContent = time === null ? "--" : `${fmtTime(time)} (${time.toFixed(6)}s)`;
}

function updateGridMarkerUi() {
  const info = blueMarkerInfo();
  updateMarkerAcceptButton("acceptGridBtn", info.label ? `BLUE ${info.label}` : "BLUE", info.time);
  const target = $("blueMarker");
  if (target) {
    const sourceLabel =
      info.source === "als_impact"
        ? "ALS impact"
        : info.source === "candidate"
          ? "exact"
          : info.source === "grid"
            ? "grid"
            : info.source === "manual"
              ? "placed"
              : "";
    const suffix = [info.label, sourceLabel, info.barNumber ? `b${info.barNumber}` : ""].filter(Boolean).join(" ");
    target.textContent = info.time === null ? "none" : `${fmtTime(info.time)} (${info.time.toFixed(6)}s)${suffix ? ` ${suffix}` : ""}`;
  }
}

function formatDropPoint(point) {
  const time = optionalMarkerTime(point?.time_sec);
  if (time === null) return "none";
  const sampleRaw = point?.sample;
  const sample = sampleRaw === null || sampleRaw === undefined ? null : Number(sampleRaw);
  return `${fmtTime(time)} (${time.toFixed(9)}s)${Number.isInteger(sample) ? ` sample ${sample}` : ""}`;
}

function renderDropPointContract() {
  const points = dropPointContract();
  const status = $("alsAnchorStatus");
  if (!points) {
    if (status) status.textContent = "--";
    for (const id of ["gridOneMarker", "visualDropMarker", "alsImpactMarker", "diagnosticZeroMarker", "boomCrestMarker", "alternateAttackMarkers"]) {
      if ($(id)) $(id).textContent = "none";
    }
    if ($("phaseTranslation")) $("phaseTranslation").textContent = "--";
    return;
  }
  if (status) {
    status.textContent = points.status === "accepted" ? "READY — ALS impact is blue" : `HOLD — ${points.reason || "anchor proof failed"}`;
    status.className = `anchorStatus ${points.status === "accepted" ? "accepted" : "rejected"}`;
  }
  if ($("gridOneMarker")) $("gridOneMarker").textContent = formatDropPoint(points.musical_one);
  if ($("visualDropMarker")) $("visualDropMarker").textContent = formatDropPoint(points.visual_marker);
  if ($("alsImpactMarker")) $("alsImpactMarker").textContent = formatDropPoint(points.als_impact);
  if ($("diagnosticZeroMarker")) $("diagnosticZeroMarker").textContent = formatDropPoint(points.zero_crossing);
  if ($("boomCrestMarker")) {
    const crest = formatDropPoint(points.body_crest);
    const crestOffsetRaw = points.body_crest_evidence?.crest_offset_ms;
    const crestOffset = crestOffsetRaw === null || crestOffsetRaw === undefined ? null : Number(crestOffsetRaw);
    $("boomCrestMarker").textContent = `${crest}${crest !== "none" && Number.isFinite(crestOffset) ? ` (+${crestOffset.toFixed(3)}ms, diagnostic)` : ""}`;
  }
  const phaseSamplesRaw = points.phase_translation?.samples;
  const phaseMsRaw = points.phase_translation?.milliseconds;
  const phaseSamples = phaseSamplesRaw === null || phaseSamplesRaw === undefined ? null : Number(phaseSamplesRaw);
  const phaseMs = phaseMsRaw === null || phaseMsRaw === undefined ? null : Number(phaseMsRaw);
  if ($("phaseTranslation")) {
    $("phaseTranslation").textContent = Number.isFinite(phaseSamples) || Number.isFinite(phaseMs)
      ? `${Number.isFinite(phaseSamples) ? `${phaseSamples} samples` : "--"}${Number.isFinite(phaseMs) ? ` / ${phaseMs.toFixed(3)} ms` : ""}`
      : "--";
  }
  if ($("alternateAttackMarkers")) {
    const alternatives = alternateAttackPoints();
    $("alternateAttackMarkers").textContent = alternatives.length
      ? alternatives.map((row, index) => `A${index + 1} ${formatDropPoint(row)}${Number.isFinite(Number(row.confidence)) ? ` c${Number(row.confidence).toFixed(3)}` : ""}`).join(" | ")
      : "none";
  }
}

function updateMetric(id, value, formatter) {
  const target = $(id);
  if (!target) return;
  const numeric = Number(value);
  target.textContent = Number.isFinite(numeric) ? formatter(numeric) : "--";
}

function updateSaveButton() {
  const micro = selectedMicroInfo();
  const times = markerTimes();
  const saveButton = $("saveCorrectionBtn");
  const hasPick = userPick !== null;
  const busy = reviewUiBusy();
  if (!saveButton) return;
  saveButton.disabled = !hasPick || busy;
  saveButton.classList.toggle("disabled", !hasPick || busy);
  saveButton.textContent = saveInFlight
    ? "SAVING..."
    : times.manual === null
      ? "SAVE PLACED"
      : `SAVE ${fmtTime(times.manual)}`;
  if ($("clearMarkerBtn")) {
    $("clearMarkerBtn").disabled = !hasPick || busy;
    $("clearMarkerBtn").classList.toggle("disabled", !hasPick || busy);
  }
  if ($("skipBtn")) {
    $("skipBtn").disabled = busy;
    $("skipBtn").classList.toggle("disabled", busy);
    $("skipBtn").textContent = busy ? "WORKING..." : "SKIP";
  }
  updateGridMarkerUi();
  updateAiPickDisplay();
  if ($("userPick")) $("userPick").textContent = times.manual === null ? "none" : `${fmtTime(times.manual)} (${times.manual.toFixed(6)}s)`;
  if ($("microMarker")) $("microMarker").textContent = times.knee === null ? "none" : `${fmtTime(times.knee)} (${times.knee.toFixed(6)}s)`;
  updateMetric("microOffset", micro?.snap_offset_ms, (value) => `${value.toFixed(2)} ms`);
  updateMetric("microConfidence", micro?.micro_confidence, (value) => value.toFixed(3));
  updateMetric("attackCleanliness", micro?.attack_cleanliness, (value) => value.toFixed(3));
  updateMetric("zeroCrossingQuality", micro?.zero_crossing_quality, (value) => value.toFixed(3));
  updatePlaceButton();
  drawWaveform();
}

function updatePlaceButton() {
  const button = $("placeBtn");
  if (!button) return;
  if (!visualFirstMode()) {
    const enabled = Boolean(currentItem) && !reviewUiBusy();
    button.disabled = !enabled;
    button.classList.toggle("disabled", !enabled);
    button.textContent = "PLACE MANUAL";
    button.title = "";
    return;
  }
  const blockReason = visualDirectPlaceBlockReason();
  const enabled = Boolean(currentItem) && !reviewUiBusy() && !blockReason;
  button.disabled = !enabled;
  button.classList.toggle("disabled", !enabled);
  button.textContent =
    placeValidationInFlight
      ? "CHECKING BOOM"
      : blockReason === "loading_boom_mask"
      ? "LOADING BOOM"
      : blockReason === "no_boom_front_edge"
        ? "NO BOOM EDGE"
        : "PLACE 1.1.1";
  button.title =
    placeValidationInFlight
      ? "Validating the Boom front edge before placing."
      : blockReason === "loading_boom_mask"
      ? "Waiting for the current waveform Boom mask."
      : blockReason === "no_boom_front_edge"
        ? "No hard Boom front edge is visible in this zoom window."
        : "";
}

function cloneCandidateForCorrection(candidate, time) {
  if (!candidate) return null;
  return {
    ...candidate,
    picked_from_candidate_list: true,
    picked_candidate_rank: candidate.rank ?? null,
    picked_candidate_time: Number(time),
  };
}

function setUserPick(value, candidate = null) {
  userPick = clampOriginalTime(Number(value));
  pickedCandidate = candidate ? cloneCandidateForCorrection(candidate, userPick) : null;
  updateSaveButton();
}

function setBlueAnchorCandidate(candidate) {
  blueAnchorCandidate = candidate ? { ...candidate } : null;
  updateGridMarkerUi();
  scheduleDrawWaveform();
}

function clearUserPick() {
  if (reviewUiBusy()) return;
  userPick = null;
  pickedCandidate = null;
  updateSaveButton();
  setStatus("Correction marker cleared.");
}

function setRefinedPick(info) {
  refinedInfo = info || null;
  refinedPick = info && info.microaligned_time !== undefined ? clampOriginalTime(Number(info.microaligned_time)) : null;
  updateSaveButton();
}

function clearRefinedPick() {
  refinedPick = null;
  refinedInfo = null;
  renderAutoAcceptGate(null);
  updateSaveButton();
}

function viewportDuration() {
  return Math.max(minViewSeconds(), Number(viewportEnd) - Number(viewportStart));
}

function pixelsPerSecond() {
  const width = $("waveWrapper").clientWidth || 1;
  return width / Math.max(viewportDuration(), minViewSeconds());
}

function timeToX(originalTime) {
  return (Number(originalTime) - viewportStart) * pixelsPerSecond();
}

function setupCanvas() {
  const canvas = $("waveCanvas");
  const rect = $("waveWrapper").getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssWidth = Math.max(1, Math.round(rect.width));
  const cssHeight = waveCanvasHeight();
  const targetWidth = Math.max(1, Math.round(cssWidth * dpr));
  const targetHeight = Math.max(1, Math.round(cssHeight * dpr));
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { canvas, ctx, width: cssWidth, height: cssHeight, dpr };
}

function chooseGridStep(secondsPerPixel) {
  const targetSeconds = Math.max(secondsPerPixel * 120, 1 / Math.max(sampleRate(), 1));
  const steps = [
    0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
    0.001, 0.002, 0.005,
    0.01, 0.02, 0.05,
    0.1, 0.25, 0.5,
    1, 2, 5,
    10, 15, 30,
    60, 120, 300,
    600,
  ];
  return steps.find((step) => step >= targetSeconds) || steps[steps.length - 1];
}

function drawGrid(ctx, width, height) {
  const secondsPerPixel = viewportDuration() / Math.max(width, 1);
  const step = chooseGridStep(secondsPerPixel);
  const majorEvery = step < 0.001 ? 10 : step < 0.01 ? 5 : step < 1 ? 4 : 2;
  let t = Math.floor(viewportStart / step) * step;
  let count = 0;
  ctx.font = "10px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  while (t <= viewportEnd + step && count < 300) {
    const x = timeToX(t);
    if (x >= -40 && x <= width + 40) {
      const idx = Math.round(t / step);
      const major = idx % majorEvery === 0;
      ctx.strokeStyle = major ? "rgba(23,32,42,0.18)" : "rgba(23,32,42,0.08)";
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, 0);
      ctx.lineTo(Math.round(x) + 0.5, height);
      ctx.stroke();
      if (major) {
        ctx.fillStyle = "rgba(23,32,42,0.58)";
        ctx.fillText(fmtTime(t), x + 4, 14);
      }
    }
    t += step;
    count += 1;
  }
}

function drawPhraseGuides(ctx, width, height) {
  const bpm = Number(currentItem?.bpm || 0);
  if (!Number.isFinite(bpm) || bpm <= 0) return;
  const barSeconds = 4 * 60 / bpm;
  if (!Number.isFinite(barSeconds) || barSeconds <= 0) return;
  const barZero = barZeroSeconds();
  const guideBars = 2;
  const guideSeconds = guideBars * barSeconds;
  const startIndex = Math.max(1, Math.ceil((viewportStart - barZero) / guideSeconds));
  const endIndex = Math.floor((viewportEnd - barZero) / guideSeconds);
  const pxSpacing = guideSeconds * pixelsPerSecond();
  ctx.save();
  ctx.font = "10px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  for (let index = startIndex; index <= endIndex && index - startIndex < 420; index += 1) {
    const bars = index * guideBars;
    const time = barZero + (index * guideSeconds);
    const x = timeToX(time);
    if (x < -40 || x > width + 40) continue;
    const is32 = bars % 32 === 0;
    const is24 = bars % 24 === 0;
    const is16 = bars % 16 === 0;
    const is8 = bars % 8 === 0;
    const is4 = bars % 4 === 0;
    ctx.strokeStyle = is32
      ? "rgba(194,53,43,0.56)"
      : is24
        ? "rgba(174,101,18,0.52)"
        : is16
          ? "rgba(15,111,191,0.45)"
          : is8
            ? "rgba(16,128,103,0.34)"
            : is4
              ? "rgba(23,32,42,0.20)"
              : "rgba(23,32,42,0.12)";
    ctx.lineWidth = is32 || is24 ? 2 : is16 ? 1.65 : is8 ? 1.35 : 1;
    ctx.setLineDash(is4 || is8 || is16 || is24 || is32 ? [] : [3, 5]);
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, 0);
    ctx.lineTo(Math.round(x) + 0.5, height);
    ctx.stroke();
    const showDense = pxSpacing >= 70;
    const showMid = pxSpacing >= 42 && (is4 || is8);
    const showSparse = is32 || is24 || is16 || (pxSpacing >= 22 && is8);
    if (showSparse || showMid || showDense) {
      const label = `${bars} bars`;
      const labelWidth = Math.ceil(ctx.measureText(label).width) + 8;
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.fillRect(x + 4, height - 20, labelWidth, 15);
      ctx.fillStyle = is32 ? "#a4261d" : is24 ? "#95580c" : is16 ? "#0a4e88" : is8 ? "#0b6651" : "rgba(23,32,42,0.64)";
      ctx.fillText(label, x + 8, height - 9);
    }
  }
  ctx.restore();
}

function drawBpmClock(ctx, width, height) {
  const bpm = Number(currentItem?.bpm || 0);
  if (!Number.isFinite(bpm) || bpm <= 0) return;
  const beatSeconds = 60 / bpm;
  if (!Number.isFinite(beatSeconds) || beatSeconds <= 0) return;
  const barZero = barZeroSeconds();
  const beatPx = beatSeconds * pixelsPerSecond();
  const firstBeat = Math.max(0, Math.floor((viewportStart - barZero) / beatSeconds) - 1);
  const lastBeat = Math.ceil((viewportEnd - barZero) / beatSeconds) + 1;
  const maxBeats = 1400;
  if (lastBeat - firstBeat > maxBeats) return;

  ctx.save();
  ctx.font = "10px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  const subtleGuides = visualFirstMode();
  for (let beat = firstBeat; beat <= lastBeat; beat += 1) {
    const time = barZero + beat * beatSeconds;
    const x = timeToX(time);
    if (x < -30 || x > width + 30) continue;
    const isOne = beat % 4 === 0;
    const isPhrase = beat % 64 === 0;
    const isEightBar = beat % 32 === 0;
    ctx.setLineDash([]);
    ctx.strokeStyle = subtleGuides
      ? isPhrase
        ? "rgba(15,111,191,0.30)"
        : isEightBar
          ? "rgba(15,111,191,0.22)"
          : isOne
            ? "rgba(15,111,191,0.16)"
            : "rgba(15,111,191,0.05)"
      : isPhrase
        ? "rgba(15,111,191,0.62)"
        : isEightBar
          ? "rgba(15,111,191,0.45)"
          : isOne
            ? "rgba(15,111,191,0.32)"
            : "rgba(15,111,191,0.12)";
    ctx.lineWidth = subtleGuides
      ? isPhrase ? 1.6 : isEightBar ? 1.35 : isOne ? 1 : 0.55
      : isPhrase ? 2.2 : isEightBar ? 1.8 : isOne ? 1.35 : 0.75;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, 0);
    ctx.lineTo(Math.round(x) + 0.5, height);
    ctx.stroke();

    if (!subtleGuides && isOne && (beatPx >= 42 || isPhrase || isEightBar)) {
      const barNumber = Math.floor(beat / 4) + 1;
      const label = `${Math.round(bpm)} BPM ONE b${barNumber}`;
      const labelWidth = Math.ceil(ctx.measureText(label).width) + 8;
      ctx.fillStyle = "rgba(255,255,255,0.88)";
      ctx.fillRect(x + 4, 24, labelWidth, 15);
      ctx.fillStyle = "#0a4e88";
      ctx.fillText(label, x + 8, 35);
    } else if (!subtleGuides && !isOne && beatPx >= 90) {
      const beatInBar = (beat % 4) + 1;
      ctx.fillStyle = "rgba(15,111,191,0.58)";
      ctx.fillText(String(beatInBar), x + 4, 35);
    }
  }
  ctx.restore();
}

function stemStroke(tile) {
  return STEM_STROKES[String(tile?.role || "drums")] || "#263746";
}

function stemFill(tile) {
  return STEM_FILLS[String(tile?.role || "drums")] || "rgba(38, 55, 70, 0.12)";
}

function rgbaFromHex(hex, alpha) {
  const value = String(hex || "#263746").replace("#", "");
  const normalized = value.length === 3
    ? value.split("").map((char) => char + char).join("")
    : value.padEnd(6, "0").slice(0, 6);
  const intValue = Number.parseInt(normalized, 16);
  const r = (intValue >> 16) & 255;
  const g = (intValue >> 8) & 255;
  const b = intValue & 255;
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, Number(alpha) || 0))})`;
}

function waveformGain(tile = waveTile) {
  return 0.92 / Math.max(Number(tile?.amplitude_peak || 1), 1e-9);
}

function sampleY(sample, laneTop, laneHeight, tile = waveTile) {
  return laneTop + laneHeight / 2 - Math.max(-1, Math.min(1, Number(sample) * waveformGain(tile))) * (laneHeight * 0.40);
}

function scopedTileMetric(tile, key, scope = "local") {
  if (scope === "global") {
    const globalValue = Number(tile?.[`global_${key}`]);
    if (Number.isFinite(globalValue) && globalValue > 0) return globalValue;
  }
  const localValue = Number(tile?.[key]);
  if (Number.isFinite(localValue) && localValue > 0) return localValue;
  return null;
}

function rmsVisualCeiling(tile = waveTile, scope = "local") {
  const ceiling = Number(
    scopedTileMetric(tile, "rms_visual_ceiling", scope) ||
      scopedTileMetric(tile, "rms_percentile_95", scope) ||
      scopedTileMetric(tile, "rms_percentile_99", scope) ||
      scopedTileMetric(tile, "rms_peak", scope) ||
      0,
  );
  if (Number.isFinite(ceiling) && ceiling > 1e-6) return ceiling;
  const rms = Array.isArray(tile?.rms) ? tile.rms.map(Number).filter((value) => Number.isFinite(value) && value > 0) : [];
  if (!rms.length) return Math.max(Number(tile?.amplitude_peak || 1), 1e-6);
  rms.sort((a, b) => a - b);
  return Math.max(rms[Math.floor(rms.length * 0.95)] || rms[rms.length - 1] || 1, 1e-6);
}

function rmsMaximizerMakeup(tile = waveTile, scope = "local") {
  const explicit = Number(scope === "global" ? tile?.global_visual_maximizer_makeup_gain : tile?.visual_maximizer_makeup_gain);
  if (Number.isFinite(explicit) && explicit > 0) {
    return Math.max(HYPER_RMS_MIN_MAKEUP, Math.min(HYPER_RMS_MAX_MAKEUP, explicit));
  }
  const p95 = Number(scopedTileMetric(tile, "rms_percentile_95", scope) || rmsVisualCeiling(tile, scope));
  const reference = Math.max(p95, 1e-6);
  return Math.max(HYPER_RMS_MIN_MAKEUP, Math.min(HYPER_RMS_MAX_MAKEUP, HYPER_RMS_TARGET / reference));
}

function softLimitMaximizedRms(value, tile = waveTile, scope = "local") {
  const makeup = rmsMaximizerMakeup(tile, scope);
  const ceiling = Math.max(
    0.5,
    Math.min(1, Number(scope === "global" ? tile?.global_visual_maximizer_ceiling : tile?.visual_maximizer_ceiling) || HYPER_RMS_CEILING),
  );
  const knee = Math.max(
    0.2,
    Math.min(ceiling - 0.02, Number(scope === "global" ? tile?.global_visual_maximizer_knee : tile?.visual_maximizer_knee) || HYPER_RMS_KNEE),
  );
  const lifted = Math.max(0, Number(value) || 0) * makeup;
  if (lifted <= knee) return lifted;
  const range = Math.max(0.001, ceiling - knee);
  return knee + range * Math.tanh((lifted - knee) / range);
}

function hyperRmsAmplitude(value, tile = waveTile, scope = "local") {
  const limited = softLimitMaximizedRms(value, tile, scope);
  const ceiling = Math.max(
    0.5,
    Math.min(1, Number(scope === "global" ? tile?.global_visual_maximizer_ceiling : tile?.visual_maximizer_ceiling) || HYPER_RMS_CEILING),
  );
  return Math.max(0, Math.min(1, Math.pow(limited / ceiling, HYPER_RMS_POWER)));
}

function boomRmsAmplitude(value, tile = waveTile) {
  return hyperRmsAmplitude(value, tile, tile?.global_visual_maximizer_makeup_gain ? "global" : "local");
}

function smoothRmsValues(values, binSpan) {
  const source = values.map((value) => Math.max(0, Number(value) || 0));
  if (source.length < 3) return source;
  const targetSeconds = Math.max(
    HYPER_RMS_MIN_SMOOTH_SECONDS,
    Math.min(HYPER_RMS_MAX_SMOOTH_SECONDS, viewportDuration() / 500),
  );
  const radius = Math.max(1, Math.round(targetSeconds / Math.max(binSpan, 1e-9) / 2));
  if (radius <= 1) return source;
  const prefix = new Array(source.length + 1).fill(0);
  for (let i = 0; i < source.length; i += 1) prefix[i + 1] = prefix[i] + source[i];
  return source.map((value, index) => {
    const i0 = Math.max(0, index - radius);
    const i1 = Math.min(source.length, index + radius + 1);
    const averaged = (prefix[i1] - prefix[i0]) / Math.max(1, i1 - i0);
    return Math.max(averaged, value * 0.18);
  });
}

function percentileValue(values, percentile) {
  const finite = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!finite.length) return 0;
  const index = Math.max(0, Math.min(finite.length - 1, Math.round((finite.length - 1) * Number(percentile))));
  return finite[index];
}

function movingAverage(values, radius) {
  const safeRadius = Math.max(0, Math.round(Number(radius) || 0));
  if (safeRadius <= 0 || values.length < 3) return values.slice();
  const prefix = new Array(values.length + 1).fill(0);
  for (let i = 0; i < values.length; i += 1) prefix[i + 1] = prefix[i] + Number(values[i] || 0);
  return values.map((_value, index) => {
    const start = Math.max(0, index - safeRadius);
    const end = Math.min(values.length, index + safeRadius + 1);
    return (prefix[end] - prefix[start]) / Math.max(1, end - start);
  });
}

function isDrumsTile(tile) {
  const role = String(tile?.role || "").toLowerCase();
  const label = String(tile?.label || "").toLowerCase();
  return role === "drums" || label.includes("drum");
}

function boomRoleTuning(tile) {
  const drums = isDrumsTile(tile);
  return {
    floorBlend: drums ? BOOM_BODY_FLOOR_BLEND : 0.82,
    minimumNormalized: drums ? BOOM_BODY_MIN_NORMALIZED : BOOM_BODY_SIDE_MIN_NORMALIZED,
    chunkMinimumNormalized: drums ? 0.280 : 0.430,
    alphaScale: drums ? 1.0 : 0.62,
    floorLift: drums ? 1.0 : 1.08,
  };
}

function scopedBoomMetric(tile, key) {
  return scopedTileMetric(tile, key, tile?.global_visual_maximizer_makeup_gain ? "global" : "local");
}

function boomAmplitudeMetric(tile, key) {
  const rawValue = scopedBoomMetric(tile, key);
  if (rawValue === null) return null;
  const amplitude = boomRmsAmplitude(rawValue, tile);
  return Number.isFinite(amplitude) ? amplitude : null;
}

function boomBodySeries(tile, rawRms, binSpan) {
  const energy = rawRms.map((value) => boomRmsAmplitude(value, tile));
  const bodyRadius = Math.max(1, Math.round((BOOM_BODY_SMOOTH_SECONDS / Math.max(binSpan, 1e-9)) / 2));
  return {
    energy,
    body: movingAverage(energy, bodyRadius),
  };
}

function boomVisualStats(tile, rawRms = null, binSpanValue = null) {
  const values = Array.isArray(rawRms) ? rawRms : Array.isArray(tile?.rms) ? tile.rms : [];
  if (values.length < 2) return null;
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = Number(binSpanValue) > 0 ? Number(binSpanValue) : span / Math.max(1, values.length);
  const { body, energy } = boomBodySeries(tile, values, binSpan);
  const q55 = percentileValue(body, 0.55);
  const q70 = percentileValue(body, 0.70);
  const q75 = percentileValue(body, 0.75);
  const q84 = percentileValue(body, 0.84);
  const q90 = percentileValue(body, 0.90);
  const q94 = percentileValue(body, 0.94);
  const q96 = percentileValue(body, BOOM_BODY_CEILING_PERCENTILE);
  const tuning = boomRoleTuning(tile);
  const globalFloor = boomAmplitudeMetric(tile, "rms_boom_floor");
  const globalCeiling = boomAmplitudeMetric(tile, "rms_boom_ceiling");
  const floor = Math.max(
    BOOM_BODY_NOISE_FLOOR,
    (globalFloor || 0) * tuning.floorLift,
    q55 + Math.max(0, q90 - q55) * tuning.floorBlend,
    q70 * (isDrumsTile(tile) ? 0.98 : 1.08),
    q84 * (isDrumsTile(tile) ? 0.82 : 0.96),
    q94 * (isDrumsTile(tile) ? 0.52 : 0.68),
  );
  const ceiling = Math.max(floor + 0.045, globalCeiling || 0, q96, q94, q90);
  return {
    energy,
    body,
    binSpan,
    floor,
    ceiling,
    q75,
    q84,
    q90,
    q94,
    tuning,
  };
}

function boomFrontEdgeWindowOk(stats, index, binSpan) {
  if (!stats || !Array.isArray(stats.energy) || !stats.energy.length) return false;
  const safeIndex = Math.max(0, Math.min(stats.energy.length - 1, Math.round(Number(index) || 0)));
  const edgeGraceBins = Math.max(1, Math.round(BOOM_PLACE_EDGE_GRACE_SECONDS / Math.max(Number(binSpan) || 0, 1e-9)));
  const minimumRawDensity = Math.min(0.280, BOOM_BODY_MIN_NORMALIZED * 0.78);
  const hardMask = stats.energy.map((value) => {
    const energy = Number(value) || 0;
    const rawDensity = normalizeBoomDensity(energy, stats);
    return energy >= stats.floor && (rawDensity >= minimumRawDensity || energy >= stats.q90);
  });
  const bridgeGapBins = Math.max(1, Math.round(0.035 / Math.max(Number(binSpan) || 0, 1e-9)));
  let cursor = 0;
  while (cursor < hardMask.length) {
    if (hardMask[cursor]) {
      cursor += 1;
      continue;
    }
    const gapStart = cursor;
    while (cursor < hardMask.length && !hardMask[cursor]) cursor += 1;
    const gapEnd = cursor;
    const leftActive = gapStart > 0 && hardMask[gapStart - 1];
    const rightActive = gapEnd < hardMask.length && hardMask[gapEnd];
    if (leftActive && rightActive && gapEnd - gapStart <= bridgeGapBins) {
      for (let fillIndex = gapStart; fillIndex < gapEnd; fillIndex += 1) hardMask[fillIndex] = true;
    }
  }
  let runStart = null;
  for (let i = safeIndex; i >= 0; i -= 1) {
    if (!hardMask[i]) break;
    runStart = i;
  }
  return runStart !== null && safeIndex <= runStart + edgeGraceBins;
}

function normalizeBoomDensity(value, stats) {
  if (!stats) return 0;
  return Math.max(0, Math.min(1, (Number(value || 0) - stats.floor) / Math.max(stats.ceiling - stats.floor, 0.001)));
}

function primaryDrumsTile() {
  return usableWaveTiles().find((tile) => isDrumsTile(tile)) || (isDrumsTile(waveTile) ? waveTile : null);
}

function serverBoomMaskAllows(tile, time, maskKey = "boom_placeable_mask", radiusBins = 1) {
  const mask = Array.isArray(tile?.[maskKey]) ? tile[maskKey] : null;
  if (!mask || !mask.length) return null;
  const start = tileStartSec(tile);
  const end = tileEndSec(tile);
  if (time < start || time > end) return false;
  const span = Math.max(end - start, 1 / Math.max(sampleRate(), 1));
  const index = Math.max(0, Math.min(mask.length - 1, Math.floor(((time - start) / span) * mask.length)));
  const radius = Math.max(0, Math.round(Number(radiusBins) || 0));
  for (let offset = -radius; offset <= radius; offset += 1) {
    const testIndex = index + offset;
    if (testIndex >= 0 && testIndex < mask.length && Boolean(mask[testIndex])) return true;
  }
  return false;
}

function closestServerBoomMaskTime(tile, time, maskKey = "boom_placeable_mask") {
  const mask = Array.isArray(tile?.[maskKey]) ? tile[maskKey].map(Boolean) : null;
  if (!mask || !mask.length) return null;
  const start = tileStartSec(tile);
  const end = tileEndSec(tile);
  const span = Math.max(end - start, 1 / Math.max(sampleRate(), 1));
  const target = Number(time);
  if (!Number.isFinite(target)) return null;
  let best = null;
  let runStart = null;
  for (let index = 0; index <= mask.length; index += 1) {
    const active = index < mask.length && Boolean(mask[index]);
    if (active && runStart === null) runStart = index;
    if (active || runStart === null) continue;
    const runEnd = index - 1;
    const edgeTime = start + (runStart / mask.length) * span;
    const runEndTime = start + ((runEnd + 1) / mask.length) * span;
    const distanceToRun = target < edgeTime ? edgeTime - target : target > runEndTime ? target - runEndTime : 0;
    const frontEdgeDistance = Math.abs(target - edgeTime);
    const row = {
      time: clampOriginalTime(edgeTime),
      runStart,
      runEnd,
      distanceToRun,
      frontEdgeDistance,
      binSpan: span / Math.max(mask.length, 1),
    };
    if (
      !best ||
      row.distanceToRun < best.distanceToRun ||
      (Math.abs(row.distanceToRun - best.distanceToRun) <= 1e-9 && row.frontEdgeDistance < best.frontEdgeDistance) ||
      (Math.abs(row.distanceToRun - best.distanceToRun) <= 1e-9 &&
        Math.abs(row.frontEdgeDistance - best.frontEdgeDistance) <= 1e-9 &&
        row.runStart < best.runStart)
    ) {
      best = row;
    }
    runStart = null;
  }
  return best;
}

function boomPlacementSnapWindowSeconds(tile = primaryDrumsTile()) {
  const span = Math.max(viewportDuration(), tile ? tileEndSec(tile) - tileStartSec(tile) : 0, 0);
  const maskLength = Array.isArray(tile?.boom_placeable_mask) ? tile.boom_placeable_mask.length : 0;
  const binSpan = maskLength > 0 && span > 0 ? span / maskLength : minViewSeconds();
  return Math.max(
    binSpan * 3,
    Math.min(BOOM_PLACE_SNAP_MAX_SECONDS, Math.max(BOOM_PLACE_EDGE_GRACE_SECONDS, span * BOOM_PLACE_SNAP_VIEW_FRACTION)),
  );
}

function boomMaskFrontEdgeSnap(time, tile = primaryDrumsTile()) {
  const snap = closestServerBoomMaskTime(tile, time, "boom_placeable_mask");
  if (!snap) return null;
  const snapWindow = boomPlacementSnapWindowSeconds(tile);
  if (snap.distanceToRun > snapWindow && snap.frontEdgeDistance > snapWindow) return null;
  const evidence = boomPlacementEvidence(snap.time, tile);
  return {
    ...snap,
    evidence,
    ok: Boolean(evidence.ok),
    snapWindow,
  };
}

function resolveBoomPlacementTime(rawTime) {
  const target = clampOriginalTime(rawTime);
  if (!visualFirstMode()) return { ok: true, time: target, rawTime: target, snapped: false, evidence: { ok: true } };
  const tile = primaryDrumsTile();
  const rawEvidence = boomPlacementEvidence(target, tile);
  const snap = boomMaskFrontEdgeSnap(target, tile);
  if (snap?.ok) {
    return {
      ok: true,
      time: snap.time,
      rawTime: target,
      snapped: Math.abs(Number(snap.time) - Number(target)) > Math.max(minViewSeconds(), snap.binSpan || 0),
      evidence: snap.evidence,
      snap,
    };
  }
  if (rawEvidence.ok) {
    return { ok: true, time: target, rawTime: target, snapped: false, evidence: rawEvidence, snap };
  }
  return { ok: false, time: target, rawTime: target, snapped: false, evidence: rawEvidence, snap };
}

async function validateVisualPlacementOnServer(markerTime, options = {}) {
  if (!visualFirstMode() || !currentItem) return { valid: true };
  const context = String(options.context || "manual");
  const allowBlueContext = context === "blue";
  const explicitCandidate = options.candidate && typeof options.candidate === "object" ? options.candidate : null;
  const candidateForValidation = allowBlueContext
    ? explicitCandidate || blueAnchorCandidate || currentItem.selected_candidate || null
    : null;
  const pickedCandidatePayload = candidateForValidation ? cloneCandidateForCorrection(candidateForValidation, markerTime) : null;
  return fetchJson("/api/validate_visual_marker", {
    method: "POST",
    body: JSON.stringify({
      id: currentItem.id,
      marker_time: markerTime,
      context,
      picked_candidate: pickedCandidatePayload,
    }),
  });
}

function serverBoomSeries(tile, expectedLength = 0) {
  const density = Array.isArray(tile?.boom_body_density)
    ? tile.boom_body_density.map((value) => Math.max(0, Math.min(1, Number(value) || 0)))
    : [];
  const bodyMask = Array.isArray(tile?.boom_body_mask) ? tile.boom_body_mask.map(Boolean) : [];
  const relevantMask = Array.isArray(tile?.boom_relevant_mask) ? tile.boom_relevant_mask.map(Boolean) : [];
  const placeableMask = Array.isArray(tile?.boom_placeable_mask) ? tile.boom_placeable_mask.map(Boolean) : [];
  const length = Math.max(density.length, bodyMask.length, relevantMask.length, placeableMask.length);
  if (length < 2) return null;
  const expected = Math.max(0, Math.round(Number(expectedLength) || 0));
  if (expected > 0 && Math.abs(length - expected) > Math.max(2, expected * 0.02)) return null;
  const normalizedDensity = Array.from({ length }, (_value, index) => density[index] || 0);
  const normalizedBodyMask = Array.from({ length }, (_value, index) => Boolean(bodyMask[index]));
  const normalizedRelevantMask = Array.from({ length }, (_value, index) => {
    if (relevantMask.length) return Boolean(relevantMask[index]);
    return Boolean(placeableMask[index]) || (Boolean(bodyMask[index]) && (density[index] || 0) >= BOOM_GUI_BODY_VISIBLE_DENSITY);
  });
  const normalizedPlaceableMask = Array.from({ length }, (_value, index) => Boolean(placeableMask[index]));
  return {
    density: normalizedDensity,
    bodyMask: normalizedBodyMask,
    relevantMask: normalizedRelevantMask,
    placeableMask: normalizedPlaceableMask,
    hasBody: normalizedBodyMask.some(Boolean),
    hasRelevant: normalizedRelevantMask.some(Boolean),
    hasPlaceable: normalizedPlaceableMask.some(Boolean),
  };
}

function serverBoomDisplayActive(serverSeries, index, tile = waveTile) {
  if (!serverSeries) return false;
  const safeIndex = Math.max(0, Math.min(serverSeries.density.length - 1, Math.round(Number(index) || 0)));
  if (serverSeries.placeableMask[safeIndex]) return true;
  return Boolean(serverSeries.relevantMask[safeIndex]);
}

function serverBoomInspectionActiveAtTime(tile, time, serverSeries = null, radiusBins = 1) {
  const series = serverSeries || serverBoomSeries(tile);
  if (!series) return !visualFirstMode();
  const start = tileStartSec(tile);
  const end = tileEndSec(tile);
  if (time < start || time > end) return false;
  const span = Math.max(end - start, 1 / Math.max(sampleRate(), 1));
  const index = Math.max(0, Math.min(series.density.length - 1, Math.floor(((time - start) / span) * series.density.length)));
  const radius = Math.max(0, Math.round(Number(radiusBins) || 0));
  for (let offset = -radius; offset <= radius; offset += 1) {
    const testIndex = index + offset;
    if (testIndex >= 0 && testIndex < series.density.length && serverBoomDisplayActive(series, testIndex, tile)) {
      return true;
    }
  }
  return false;
}

function serverBoomRunStarts(mask) {
  if (!Array.isArray(mask) || !mask.length) return [];
  const starts = [];
  let active = false;
  mask.forEach((value, index) => {
    const on = Boolean(value);
    if (on && !active) starts.push(index);
    active = on;
  });
  return starts;
}

function serverBoomActiveRuns(serverSeries, length, tile = waveTile) {
  const safeLength = Math.max(0, Math.round(Number(length) || 0));
  if (!serverSeries || safeLength <= 0) return [];
  const runs = [];
  let startIndex = null;
  for (let index = 0; index <= safeLength; index += 1) {
    const active = index < safeLength && serverBoomDisplayActive(serverSeries, index, tile);
    if (active && startIndex === null) {
      startIndex = index;
    } else if (!active && startIndex !== null) {
      runs.push({ startIndex, endIndex: index });
      startIndex = null;
    }
  }
  return runs;
}

function drawBoomMaskedCenterline(ctx, width, mid, serverSeries, length, tile, start, binSpan) {
  if (!serverSeries || !Number.isFinite(binSpan) || binSpan <= 0) return;
  ctx.save();
  ctx.strokeStyle = "rgba(23,32,42,0.16)";
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 5]);
  serverBoomActiveRuns(serverSeries, length, tile).forEach((run) => {
    const x0 = Math.max(0, timeToX(start + run.startIndex * binSpan));
    const x1 = Math.min(width, timeToX(start + run.endIndex * binSpan));
    if (x1 - x0 < 2) return;
    ctx.beginPath();
    ctx.moveTo(x0, Math.round(mid) + 0.5);
    ctx.lineTo(x1, Math.round(mid) + 0.5);
    ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.restore();
}

function boomPlacementEvidence(time, tile = primaryDrumsTile()) {
  const rawRms = Array.isArray(tile?.rms) ? tile.rms : [];
  if (!visualFirstMode()) return { ok: true, reason: "no_visual_gate" };
  if (!tile) return { ok: false, reason: "no_drum_boom_tile", normalizedPost: 0, normalizedPeak: 0 };
  if (!isDrumsTile(tile)) return { ok: false, reason: "not_drum_stem", normalizedPost: 0, normalizedPeak: 0 };
  if (rawRms.length < 4) return { ok: false, reason: "insufficient_boom_bins", normalizedPost: 0, normalizedPeak: 0 };
  const start = tileStartSec(tile);
  const end = tileEndSec(tile);
  if (time < start || time > end) return { ok: false, reason: "outside_current_boom_tile", normalizedPost: 0, normalizedPeak: 0 };
  const serverMaskAllowed = serverBoomMaskAllows(tile, time, "boom_placeable_mask", 2);
  if (serverMaskAllowed === null) {
    return { ok: false, reason: "missing_server_boom_placeable_mask", normalizedPost: 0, normalizedPeak: 0 };
  }
  if (serverMaskAllowed === false) {
    return { ok: false, reason: "server_boom_mask_not_placeable", normalizedPost: 0, normalizedPeak: 0 };
  }
  const span = Math.max(end - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = span / Math.max(1, rawRms.length);
  const stats = boomVisualStats(tile, rawRms, binSpan);
  if (!stats) return { ok: true, reason: "no_stats" };
  const markerIndex = Math.max(0, Math.min(stats.body.length - 1, Math.floor((time - start) / Math.max(binSpan, 1e-9))));
  const preStart = Math.max(0, Math.floor((time - BOOM_PLACE_PRE_SECONDS - start) / Math.max(binSpan, 1e-9)));
  const preEnd = Math.max(preStart, Math.min(stats.body.length, Math.floor((time - start) / Math.max(binSpan, 1e-9))));
  const postStart = markerIndex;
  const postEnd = Math.min(stats.body.length, Math.ceil((time + BOOM_PLACE_POST_SECONDS - start) / Math.max(binSpan, 1e-9)));
  const preValues = stats.body.slice(preStart, preEnd);
  const postValues = stats.body.slice(postStart, postEnd);
  if (!postValues.length) return { ok: false, reason: "no_post_body", normalizedPost: 0, normalizedPeak: 0 };
  const preFloor = preValues.length ? percentileValue(preValues, 0.58) : percentileValue(stats.body, 0.34);
  const postBody = percentileValue(postValues, 0.72);
  const postPeak = Math.max(...postValues);
  const normalizedPost = normalizeBoomDensity(postBody, stats);
  const normalizedPeak = normalizeBoomDensity(postPeak, stats);
  const contrast = Math.max(0, postBody - preFloor);
  const hardBodyFloor = postBody >= stats.floor && postPeak >= stats.floor + BOOM_PLACE_MIN_CONTRAST * 0.50;
  const enoughBody = normalizedPost >= BOOM_PLACE_MIN_POST_NORMALIZED || postBody >= stats.q90;
  const enoughPeak = normalizedPeak >= BOOM_PLACE_MIN_PEAK_NORMALIZED || postPeak >= stats.q94;
  const enoughContrast =
    contrast >= BOOM_PLACE_MIN_CONTRAST ||
    (postBody >= stats.q90 && normalizedPost >= BOOM_PLACE_MIN_POST_NORMALIZED + 0.08);
  const edgeWindow = boomFrontEdgeWindowOk(stats, markerIndex, binSpan);
  let reason = "boom_body_after_marker";
  if (!edgeWindow) reason = "not_boom_front_edge";
  else if (!hardBodyFloor) reason = "below_global_boom_floor";
  else if (!enoughBody) reason = "low_post_body";
  else if (!enoughPeak) reason = "weak_post_peak";
  else if (!enoughContrast) reason = "low_contrast";
  return {
    ok: edgeWindow && hardBodyFloor && enoughBody && enoughPeak && enoughContrast,
    reason,
    normalizedPost,
    normalizedPeak,
    contrast,
    postBody,
    postPeak,
    floor: stats.floor,
  };
}

function tileHasPlaceableBoomEvidence(tile = primaryDrumsTile()) {
  if (!visualFirstMode() || !tile || !isDrumsTile(tile)) return false;
  const serverPlaceableCount = Number(tile?.boom_placeable_count);
  if (Number.isFinite(serverPlaceableCount) && serverPlaceableCount <= 0) return false;
  if (Array.isArray(tile?.boom_placeable_mask)) return tile.boom_placeable_mask.some(Boolean);
  return false;
}

function selectedDropTime() {
  return optionalMarkerTime(currentItem?.ai_pick) || candidateTime(currentItem?.selected_candidate);
}

function valuesInTimeRange(values, start, binSpan, rangeStart, rangeEnd) {
  const out = [];
  values.forEach((value, index) => {
    const time = start + (index + 0.5) * binSpan;
    if (time >= rangeStart && time <= rangeEnd) out.push(value);
  });
  return out;
}

function selectedDropHighlightBounds(tile, rawRms, binSpan) {
  if (!visualFirstMode() || !isDrumsTile(tile) || !Array.isArray(rawRms) || rawRms.length < 4) return null;
  const marker = selectedDropTime();
  if (marker === null) return null;
  const start = tileStartSec(tile);
  const end = tileEndSec(tile);
  if (marker < start || marker > end) return null;

  const energy = rawRms.map((value) => boomRmsAmplitude(value, tile));
  const radius = Math.max(1, Math.round((VISUAL_SELECTED_DROP_SMOOTH_SECONDS / Math.max(binSpan, 1e-9)) / 2));
  const serverSeries = serverBoomSeries(tile, rawRms.length);
  const smoothed = serverSeries ? serverSeries.density : movingAverage(energy, radius);
  const evidence = boomPlacementEvidence(marker, tile);
  const selectedProof =
    currentItem?.selected_candidate?.gui_mask_proof ||
    currentItem?.gui_mask_proof ||
    currentItem?.selected_candidate?.visual_components?.gui_mask_proof ||
    null;
  const actualBodyRelief = Boolean(
    selectedProof?.accepted_by_actual_visual_body_proof ||
      selectedProof?.actual_body_sparse_impact ||
      currentItem?.selected_candidate?.visual_components?.actual_body_boom_repair_sparse_impact,
  );
  if (!evidence.ok && !actualBodyRelief) return null;
  const markerIndex = Math.max(0, Math.min(smoothed.length - 1, Math.floor((marker - start) / Math.max(binSpan, 1e-9))));

  if (serverSeries) {
    const snap = closestServerBoomMaskTime(tile, marker, "boom_placeable_mask");
    const snapLimit = Math.max(BOOM_PLACE_EDGE_GRACE_SECONDS, (snap?.binSpan || binSpan) * 3);
    if (!actualBodyRelief && (!snap || (snap.distanceToRun > snapLimit && snap.frontEdgeDistance > snapLimit))) return null;
    const markerOnServer = serverBoomDisplayActive(serverSeries, markerIndex, tile);
    const startIndex =
      actualBodyRelief || markerOnServer || !snap
        ? markerIndex
        : Math.max(0, Math.min(smoothed.length - 1, Math.round(Number(snap.runStart) || 0)));
    const maxIndex = Math.min(
      smoothed.length - 1,
      Math.ceil((start + startIndex * binSpan + VISUAL_SELECTED_DROP_MAX_SECONDS - start) / Math.max(binSpan, 1e-9)),
    );
    const minEnd = start + startIndex * binSpan + VISUAL_SELECTED_DROP_MIN_SECONDS;
    const lowRunBins = Math.max(1, Math.round(VISUAL_SELECTED_DROP_LOW_RUN_SECONDS / Math.max(binSpan, 1e-9)));
    let endIndex = Math.max(startIndex, snap ? Math.round(Number(snap.runEnd) || startIndex) : startIndex);
    let lowRun = 0;
    for (let index = startIndex; index <= maxIndex; index += 1) {
      const time = start + (index + 1) * binSpan;
      endIndex = index;
      const serverActive = serverBoomDisplayActive(serverSeries, index, tile);
      if (!serverActive && time >= minEnd) {
        lowRun += 1;
        if (lowRun >= lowRunBins) {
          endIndex = Math.max(startIndex, index - lowRun + 1);
          break;
        }
      } else {
        lowRun = 0;
      }
    }
    const highlightStart =
      actualBodyRelief || markerOnServer || !snap
        ? marker
        : start + startIndex * binSpan;
    const stableEnd = Math.max(
      minEnd,
      Math.min(highlightStart + VISUAL_SELECTED_DROP_MAX_SECONDS, start + (endIndex + 1) * binSpan),
    );
    const densitySlice = smoothed.slice(startIndex, Math.min(smoothed.length, endIndex + 1));
    const density = densitySlice.length ? Math.max(...densitySlice) : smoothed[markerIndex] || 0;
    return {
      start: clampOriginalTime(highlightStart),
      end: Math.min(end, stableEnd),
      density,
    };
  }

  const preValues = valuesInTimeRange(smoothed, start, binSpan, marker - 0.220, marker - 0.018);
  const postValues = valuesInTimeRange(smoothed, start, binSpan, marker, marker + 0.650);
  if (!postValues.length) return null;

  const preFloor = preValues.length ? percentileValue(preValues, 0.55) : percentileValue(smoothed, 0.30);
  const postBody = percentileValue(postValues, 0.76);
  const postPeak = percentileValue(postValues, 0.92);
  const boomStats = boomVisualStats(tile, rawRms, binSpan);
  const threshold = serverSeries
    ? Math.max(0.08, Math.min(0.62, preFloor + Math.max(0.03, postBody - preFloor) * 0.34))
    : Math.max(
        0.12,
        boomStats?.floor || 0,
        preFloor + Math.max(0.03, postBody - preFloor) * 0.34,
      );
  const lowThreshold = Math.max(0.08, threshold * 0.82);
  const maxIndex = Math.min(
    smoothed.length - 1,
    Math.ceil((marker + VISUAL_SELECTED_DROP_MAX_SECONDS - start) / Math.max(binSpan, 1e-9)),
  );
  const minEnd = marker + VISUAL_SELECTED_DROP_MIN_SECONDS;
  const lowRunBins = Math.max(1, Math.round(VISUAL_SELECTED_DROP_LOW_RUN_SECONDS / Math.max(binSpan, 1e-9)));
  let endIndex = markerIndex;
  let lowRun = 0;
  for (let index = markerIndex; index <= maxIndex; index += 1) {
    const time = start + (index + 1) * binSpan;
    endIndex = index;
    const serverInactive = serverSeries && !serverSeries.bodyMask[index] && !serverSeries.placeableMask[index];
    if ((serverInactive || smoothed[index] < lowThreshold) && time >= minEnd) {
      lowRun += 1;
      if (lowRun >= lowRunBins) {
        endIndex = Math.max(markerIndex, index - lowRun + 1);
        break;
      }
    } else {
      lowRun = 0;
    }
  }

  const stableEnd = Math.max(
    minEnd,
    Math.min(marker + VISUAL_SELECTED_DROP_MAX_SECONDS, start + (endIndex + 1) * binSpan),
  );
  return {
    start: marker,
    end: Math.min(end, stableEnd),
    density: Math.max(postBody, postPeak * 0.78),
  };
}

function drawSelectedDropHighlight(ctx, width, laneTop, laneHeight, tile, bounds) {
  if (!bounds || bounds.end <= bounds.start) return;
  const x0 = Math.max(0, timeToX(bounds.start));
  const x1 = Math.min(width, timeToX(bounds.end));
  if (x1 - x0 < VISUAL_CHUNK_MIN_WIDTH_PX) return;
  const color = stemStroke(tile);
  const density = Math.max(0, Math.min(1, Number(bounds.density) || 0));
  const alpha = 0.30 + 0.24 * Math.pow(density, 0.70);
  const y = laneTop + laneHeight * 0.08;
  const h = laneHeight * 0.84;
  ctx.save();
  ctx.fillStyle = rgbaFromHex(color, alpha);
  ctx.fillRect(x0, y, x1 - x0, h);
  ctx.strokeStyle = rgbaFromHex(color, Math.min(0.72, alpha + 0.14));
  ctx.lineWidth = 1.35;
  ctx.strokeRect(x0 + 0.5, y + 0.5, Math.max(0, x1 - x0 - 1), Math.max(0, h - 1));
  ctx.restore();
}

function drawEnergyChunks(ctx, width, laneTop, laneHeight, tile = waveTile, rmsValues = null, binSpanValue = null) {
  if (!visualFirstMode() || viewportDuration() > VISUAL_CHUNK_MAX_VIEW_SECONDS) return;
  const rawRms = Array.isArray(rmsValues) ? rmsValues : Array.isArray(tile?.rms) ? tile.rms : [];
  if (rawRms.length < 4) return;
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = Number(binSpanValue) > 0 ? Number(binSpanValue) : span / Math.max(1, rawRms.length);
  const selectedBounds = selectedDropHighlightBounds(tile, rawRms, binSpan);
  const boomMode = waveformVisualMode === "boom";
  const boomStats = boomMode ? boomVisualStats(tile, rawRms, binSpan) : null;
  const serverSeries = boomMode ? serverBoomSeries(tile, rawRms.length) : null;
  if (boomMode && visualFirstMode() && !serverSeries) return;
  let smoothed = [];
  let threshold = VISUAL_CHUNK_NOISE_GATE;
  let densityCeiling = 1.0;
  let chunkMinimumNormalized = 0.18;
  let alphaScale = 1.0;
  if (boomMode && serverSeries) {
    smoothed = serverSeries.density;
    threshold = 0.0001;
    densityCeiling = 1.0;
    chunkMinimumNormalized = 0;
    alphaScale = boomRoleTuning(tile).alphaScale;
  } else if (boomMode && boomStats) {
    smoothed = boomStats.body;
    threshold = Math.max(VISUAL_CHUNK_NOISE_GATE, boomStats.floor);
    densityCeiling = Math.max(threshold + 0.001, boomStats.ceiling);
    chunkMinimumNormalized = boomStats.tuning.chunkMinimumNormalized;
    alphaScale = boomStats.tuning.alphaScale;
  } else {
    const energy = rawRms.map((value) => hyperRmsAmplitude(value, tile));
    const chunkSmoothSeconds = 0.120;
    const radius = Math.max(1, Math.round((chunkSmoothSeconds / Math.max(binSpan, 1e-9)) / 2));
    smoothed = movingAverage(energy, radius);
    const low = percentileValue(smoothed, 0.34);
    const high = percentileValue(smoothed, 0.82);
    const top = percentileValue(smoothed, 0.93);
    threshold = Math.max(
      VISUAL_CHUNK_NOISE_GATE,
      low + (Math.max(0.02, high - low) * 0.70),
    );
    densityCeiling = Math.max(threshold + 0.001, top);
  }
  const minDuration = boomMode ? 0.090 : 0.060;
  const mergeGap = boomMode ? 0.030 : 0.045;
  const chunks = [];
  let active = null;

  smoothed.forEach((value, index) => {
    const serverActive = serverSeries ? serverBoomDisplayActive(serverSeries, index, tile) : null;
    const activeBin = serverSeries ? serverActive : value >= threshold;
    if (activeBin) {
      if (!active) {
        active = {
          startIndex: index,
          endIndex: index,
          sum: 0,
          max: 0,
          maskCount: 0,
          placeableCount: 0,
          firstPlaceableIndex: null,
        };
      }
      active.endIndex = index;
      active.sum += value;
      active.max = Math.max(active.max, value);
      if (serverSeries?.bodyMask[index]) active.maskCount += 1;
      if (serverSeries?.placeableMask[index]) {
        active.placeableCount += 1;
        if (active.firstPlaceableIndex === null) active.firstPlaceableIndex = index;
      }
    } else if (active) {
      chunks.push(active);
      active = null;
    }
  });
  if (active) chunks.push(active);

  const merged = [];
  chunks.forEach((chunk) => {
    const chunkStart = start + chunk.startIndex * binSpan;
    const chunkEnd = start + (chunk.endIndex + 1) * binSpan;
    const prev = merged[merged.length - 1];
    const count = Math.max(1, chunk.endIndex - chunk.startIndex + 1);
    if (prev && chunkStart - prev.end <= mergeGap) {
      prev.end = chunkEnd;
      prev.sum += chunk.sum;
      prev.count += count;
      prev.max = Math.max(prev.max, chunk.max);
      prev.maskCount += chunk.maskCount || 0;
      prev.placeableCount += chunk.placeableCount || 0;
      if (chunk.firstPlaceableIndex !== null) {
        prev.firstPlaceableIndex =
          prev.firstPlaceableIndex === null
            ? chunk.firstPlaceableIndex
            : Math.min(prev.firstPlaceableIndex, chunk.firstPlaceableIndex);
      }
      return;
    }
    merged.push({
      start: chunkStart,
      end: chunkEnd,
      sum: chunk.sum,
      count,
      max: chunk.max,
      maskCount: chunk.maskCount || 0,
      placeableCount: chunk.placeableCount || 0,
      firstPlaceableIndex: chunk.firstPlaceableIndex,
    });
  });

  const color = stemStroke(tile);
  ctx.save();
  merged.forEach((chunk) => {
    if (
      selectedBounds &&
      chunk.end >= selectedBounds.start - 0.010 &&
      chunk.start <= selectedBounds.end + 0.010
    ) {
      return;
    }
    if (serverSeries && chunk.placeableCount <= 0) return;
    const renderStart = serverSeries && Number.isFinite(chunk.firstPlaceableIndex)
      ? start + chunk.firstPlaceableIndex * binSpan
      : chunk.start;
    const duration = chunk.end - renderStart;
    const x0 = Math.max(0, timeToX(renderStart));
    const x1 = Math.min(width, timeToX(chunk.end));
    if (duration < minDuration || x1 - x0 < VISUAL_CHUNK_MIN_WIDTH_PX) return;
    const average = chunk.sum / Math.max(1, chunk.count);
    const density = Math.max(average, chunk.max * 0.82);
    if (!serverSeries && density < Math.max(threshold, VISUAL_CHUNK_NOISE_GATE)) return;
    const normalizedDensity = serverSeries
      ? Math.max(
          density,
          chunk.placeableCount > 0 ? BOOM_PLACE_MIN_POST_NORMALIZED : BOOM_BODY_MIN_NORMALIZED,
        )
      : Math.max(0, Math.min(1, (density - threshold) / Math.max(densityCeiling - threshold, 0.001)));
    if (!serverSeries && boomMode && normalizedDensity < chunkMinimumNormalized) return;
    const frontEdgeBoost = serverSeries && chunk.placeableCount > 0 ? BOOM_GUI_FRONT_EDGE_ALPHA_BOOST : 0;
    const alpha = Math.min(
      0.76,
      alphaScale * (
        VISUAL_CHUNK_MIN_ALPHA +
        (VISUAL_CHUNK_MAX_ALPHA - VISUAL_CHUNK_MIN_ALPHA) *
          Math.pow(Math.max(0, Math.min(1, boomMode ? normalizedDensity : density)), 0.95)
      ) + frontEdgeBoost,
    );
    const y = laneTop + laneHeight * 0.08;
    const h = laneHeight * 0.84;
    ctx.fillStyle = rgbaFromHex(color, alpha);
    ctx.fillRect(x0, y, x1 - x0, h);
    ctx.strokeStyle = rgbaFromHex(color, Math.min(0.62, alpha + 0.10));
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y + 0.5, Math.max(0, x1 - x0 - 1), Math.max(0, h - 1));
  });
  ctx.restore();
  drawSelectedDropHighlight(ctx, width, laneTop, laneHeight, tile, selectedBounds);
}

function drawBoomBodyBars(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const rawRms = Array.isArray(tile?.rms) ? tile.rms : [];
  if (rawRms.length < 2) return false;
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = span / Math.max(1, rawRms.length);
  const pps = pixelsPerSecond();
  const binsPerBar = Math.max(1, Math.round(BOOM_BODY_TARGET_BAR_PX / Math.max(binSpan * pps, 0.001)));
  const boomStats = boomVisualStats(tile, rawRms, binSpan);
  if (!boomStats) return false;
  const serverSeries = serverBoomSeries(tile, rawRms.length);
  const body = serverSeries ? serverSeries.density : boomStats.body;
  const color = stemStroke(tile);
  const mid = laneTop + laneHeight / 2;

  ctx.save();
  if (visualFirstMode() && !serverSeries) {
    ctx.restore();
    return true;
  }
  if (visualFirstMode() && serverSeries) {
    drawBoomMaskedCenterline(ctx, width, mid, serverSeries, body.length, tile, start, binSpan);
  } else {
    ctx.strokeStyle = "rgba(23,32,42,0.24)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 5]);
    ctx.beginPath();
    ctx.moveTo(0, Math.round(mid) + 0.5);
    ctx.lineTo(width, Math.round(mid) + 0.5);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawEnergyChunks(ctx, width, laneTop, laneHeight, tile, rawRms, binSpan);

  for (let i = 0; i < body.length; i += binsPerBar) {
    const endIndex = Math.min(body.length, i + binsPerBar);
    const slice = body.slice(i, endIndex);
    if (!slice.length) continue;
    const serverPlaceableSlice = serverSeries ? serverSeries.placeableMask.slice(i, endIndex) : [];
    const serverPlaceable = serverPlaceableSlice.some(Boolean);
    const serverDisplay = serverSeries
      ? slice.some((_value, offset) => serverBoomDisplayActive(serverSeries, i + offset, tile))
      : true;
    if (serverSeries && !serverDisplay) continue;
    const peak = Math.max(...slice);
    const avg = slice.reduce((sum, value) => sum + value, 0) / slice.length;
    let density = Math.max(avg, peak * 0.88);
    if (serverSeries && !serverPlaceable && density < BOOM_GUI_BODY_VISIBLE_DENSITY) continue;
    const normalized = serverSeries
      ? (
          serverPlaceable
            ? Math.max(density, BOOM_PLACE_MIN_POST_NORMALIZED)
            : density
        )
      : normalizeBoomDensity(density, boomStats);
    if (!serverSeries && normalized < boomStats.tuning.minimumNormalized) continue;
    const t0 = start + i * binSpan;
    const t1 = start + endIndex * binSpan;
    const x0 = timeToX(t0);
    const x1 = timeToX(t1);
    if (x1 < -12 || x0 > width + 12) continue;
    const x = (x0 + x1) / 2;
    const barWidth = Math.max(BOOM_BODY_MIN_BAR_PX, Math.min(BOOM_BODY_MAX_BAR_PX, (x1 - x0) * 0.62));
    const amp = Math.pow(normalized, BOOM_BODY_DARK_POWER);
    const halfHeight = Math.max(1.2, amp * laneHeight * 0.43);
    const bodyContextScale =
      serverSeries && !serverPlaceable && density < BOOM_GUI_BODY_STRONG_DENSITY ? BOOM_GUI_CONTEXT_BODY_SCALE : 1.0;
    const frontEdgeScale = serverSeries && serverPlaceable ? 1.08 : 1.0;
    ctx.globalAlpha = Math.min(
      0.98,
      (0.04 + 0.92 * amp) * boomStats.tuning.alphaScale * bodyContextScale * frontEdgeScale,
    );
    ctx.strokeStyle = color;
    ctx.lineWidth = barWidth;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(x, mid - halfHeight);
    ctx.lineTo(x, mid + halfHeight);
    ctx.stroke();
  }

  if (serverSeries?.hasPlaceable) {
    const tickWidth = Math.max(1.4, Math.min(4, BOOM_GUI_FRONT_EDGE_TICK_SECONDS * pixelsPerSecond()));
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = isDrumsTile(tile) ? 0.92 : 0.42;
    ctx.lineWidth = tickWidth;
    ctx.lineCap = "butt";
    serverBoomRunStarts(serverSeries.placeableMask).forEach((index) => {
      const t = start + index * binSpan;
      const x = timeToX(t);
      if (x < -8 || x > width + 8) return;
      const density = Number(serverSeries.density[index]) || 0;
      const tickHeight = laneHeight * (0.28 + 0.42 * Math.pow(Math.max(density, BOOM_PLACE_MIN_POST_NORMALIZED), 0.72));
      ctx.beginPath();
      ctx.moveTo(x, mid - tickHeight / 2);
      ctx.lineTo(x, mid + tickHeight / 2);
      ctx.stroke();
    });
    ctx.restore();
  }

  ctx.restore();
  if (viewportDuration() <= 0.350 && tileHasPlaceableBoomEvidence(tile)) {
    drawSampleInspectionOverlay(ctx, width, laneTop, laneHeight, tile);
  }
  return true;
}

function rmsY(value, laneTop, laneHeight, tile = waveTile, polarity = 1) {
  const amp = hyperRmsAmplitude(value, tile);
  return laneTop + laneHeight / 2 - polarity * amp * (laneHeight * 0.42);
}

function waveformModeText() {
  if (waveformVisualMode === "boom") return "Global-gated boom-body bars";
  return waveformVisualMode === "rms" ? "Maximized RMS envelope" : "ultra min/max peaks";
}

function sampleInspectionAvailable(tile = waveTile) {
  const samples = tile?.samples || [];
  const spacing = pixelsPerSecond() / Math.max(Number(tile?.sample_rate || sampleRate()), 1);
  return samples.length >= 2 && spacing >= RMS_INSPECTION_MIN_SAMPLE_SPACING;
}

function sampleInspectionLabel(tile = waveTile) {
  const samples = tile?.samples || [];
  const spacing = pixelsPerSecond() / Math.max(Number(tile?.sample_rate || sampleRate()), 1);
  if (samples.length < 2 || spacing < RMS_INSPECTION_MIN_SAMPLE_SPACING) return "inspection peak";
  return spacing >= RMS_INSPECTION_ZERO_CROSSING_SPACING ? "inspection peak+sample+zero" : "inspection peak+sample";
}

function markerInspectionRadius() {
  return waveformVisualMode === "peaks" ? 0.05 : RMS_DROP_INSPECTION_RADIUS_SECONDS;
}

function setWaveformVisualMode(mode) {
  if (visualFirstMode() && mode !== "boom") {
    mode = "boom";
    setStatus("Visual-first review uses the Boom waveform only.");
  }
  waveformVisualMode = mode === "peaks" ? "peaks" : mode === "rms" ? "rms" : "boom";
  try {
    window.localStorage?.setItem(WAVEFORM_MODE_STORAGE_KEY, waveformVisualMode);
    window.localStorage?.setItem(WAVEFORM_MODE_VERSION_KEY, WAVEFORM_MODE_VERSION);
  } catch (_) {
    // Local storage can be unavailable in restricted browser contexts.
  }
  syncWaveformModeButtons();
  updateAudioStatus();
  scheduleDrawWaveform();
}

function syncWaveformModeButtons() {
  if (visualFirstMode() && waveformVisualMode !== "boom") waveformVisualMode = "boom";
  const boomButton = $("boomWaveBtn");
  const rmsButton = $("rmsWaveBtn");
  const peakButton = $("peakWaveBtn");
  boomButton?.classList.toggle("active", waveformVisualMode === "boom");
  rmsButton?.classList.toggle("active", waveformVisualMode === "rms");
  peakButton?.classList.toggle("active", waveformVisualMode === "peaks");
  [rmsButton, peakButton].forEach((button) => {
    if (!button) return;
    button.hidden = visualFirstMode();
    button.disabled = visualFirstMode();
    button.classList.toggle("disabled", visualFirstMode());
  });
}

function waveLaneSources() {
  const loaded = waveTiles.length ? waveTiles : [];
  if (loaded.length) return loaded;
  const stems = (currentItem?.stem_waveforms || []).filter((stem) => stem.available !== false);
  if (stems.length) return stems;
  if (waveTile) return [waveTile];
  return [{ role: "drums", label: "Drums" }];
}

function stemSourceForRole(role) {
  const target = String(role || "").toLowerCase();
  return waveLaneSources().find((lane) => String(lane?.role || "").toLowerCase() === target) ||
    (currentItem?.stem_waveforms || []).find((stem) => String(stem?.role || "").toLowerCase() === target) ||
    null;
}

function isStemMuted(role) {
  return Boolean(stemMuteState[String(role || "")]);
}

function setStemMuted(role, muted) {
  const key = String(role || "");
  if (!PLAYBACK_STEM_ROLES.includes(key)) return;
  stemMuteState[key] = Boolean(muted);
  if (stemMuteState[key]) stopStemPlayer(key);
  else if (audioPlayer) startStemPlayer(key, !audioPlayer.paused);
  renderStemControls();
}

function toggleStemMuted(role) {
  setStemMuted(role, !isStemMuted(role));
}

function tileStartSec(tile = waveTile) {
  if (!tile) return viewportStart;
  if (tile.start_sec !== undefined) return Number(tile.start_sec);
  if (tile.start_sample !== undefined || tile.sample_start !== undefined) {
    return Number(tile.start_sample || tile.sample_start || 0) / Math.max(sampleRate(), 1);
  }
  return viewportStart;
}

function tileEndSec(tile = waveTile) {
  if (!tile) return viewportEnd;
  if (tile.end_sec !== undefined) return Number(tile.end_sec);
  if (tile.end_sample !== undefined) return Number(tile.end_sample) / Math.max(sampleRate(), 1);
  return viewportEnd;
}

function drawLaneFrame(ctx, width, laneTop, laneHeight, lane, index) {
  ctx.fillStyle = index % 2 === 0 ? "#ffffff" : "#fbfcfd";
  ctx.fillRect(0, laneTop, width, laneHeight);
  ctx.strokeStyle = index === 0 ? "#d7dce2" : "rgba(23,32,42,0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, Math.round(laneTop) + 0.5);
  ctx.lineTo(width, Math.round(laneTop) + 0.5);
  ctx.stroke();
  const mid = laneTop + laneHeight / 2;
  ctx.strokeStyle = "rgba(23,32,42,0.14)";
  ctx.beginPath();
  ctx.moveTo(0, Math.round(mid) + 0.5);
  ctx.lineTo(width, Math.round(mid) + 0.5);
  ctx.stroke();
}

function drawLaneLabel(ctx, laneTop, lane, index) {
  const label = String(lane?.label || lane?.role || "Stem");
  ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  const labelWidth = Math.ceil(ctx.measureText(label).width) + 14;
  const labelLeft = PLAYBACK_STEM_ROLES.includes(String(lane?.role || "")) ? 60 : 8;
  ctx.fillStyle = "rgba(255,255,255,0.86)";
  ctx.fillRect(labelLeft, laneTop + 7, labelWidth, 20);
  ctx.strokeStyle = "rgba(23,32,42,0.12)";
  ctx.strokeRect(labelLeft + 0.5, laneTop + 7.5, labelWidth, 19);
  ctx.fillStyle = stemStroke(lane);
  ctx.fillText(label, labelLeft + 7, laneTop + 21);
}

function renderStemControls() {
  const overlay = $("waveControlsOverlay");
  if (!overlay) return;
  overlay.innerHTML = "";
  if (!currentItem || !waveDuration) return;
  const height = waveCanvasHeight();
  const lanes = waveLaneSources();
  const laneHeight = height / Math.max(1, lanes.length);
  lanes.forEach((lane, index) => {
    const role = String(lane?.role || "");
    if (!PLAYBACK_STEM_ROLES.includes(role) || lane?.ok === false || lane?.available === false) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `stemMuteButton${isStemMuted(role) ? " muted" : ""}`;
    button.textContent = isStemMuted(role) ? "Muted" : "Mute";
    button.title = `${isStemMuted(role) ? "Unmute" : "Mute"} ${lane?.label || role}`;
    button.style.left = "8px";
    button.style.top = `${Math.round(index * laneHeight + 7)}px`;
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      toggleStemMuted(role);
    });
    overlay.appendChild(button);
  });
}

function drawPeaks(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const mins = tile?.mins || [];
  const maxs = tile?.maxs || [];
  if (!mins.length || !maxs.length) return;
  const serverSeries = visualFirstMode() ? serverBoomSeries(tile, Math.max(mins.length, maxs.length)) : null;
  if (visualFirstMode() && !serverSeries) return;
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = span / Math.max(1, mins.length);
  const strokeWidth = Math.max(0.55, Math.min(4, binSpan * pixelsPerSecond()));
  const detailRatio = mins.length / Math.max(width, 1);
  drawEnergyChunks(ctx, width, laneTop, laneHeight, tile);
  ctx.fillStyle = stemFill(tile);
  const runs = serverSeries ? serverBoomActiveRuns(serverSeries, Math.min(mins.length, maxs.length), tile) : [
    { startIndex: 0, endIndex: Math.min(mins.length, maxs.length) },
  ];
  runs.forEach((run) => {
    if (run.endIndex <= run.startIndex) return;
    ctx.beginPath();
    for (let i = run.startIndex; i < run.endIndex; i += 1) {
      const t = start + (i + 0.5) * binSpan;
      const x = timeToX(t);
      const y = sampleY(maxs[i], laneTop, laneHeight, tile);
      if (i === run.startIndex) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = run.endIndex - 1; i >= run.startIndex; i -= 1) {
      const t = start + (i + 0.5) * binSpan;
      ctx.lineTo(timeToX(t), sampleY(mins[i], laneTop, laneHeight, tile));
    }
    ctx.closePath();
    ctx.fill();
  });

  ctx.save();
  ctx.globalAlpha = detailRatio > 2 ? 0.62 : 0.92;
  ctx.strokeStyle = stemStroke(tile);
  ctx.lineWidth = strokeWidth;
  for (let i = 0; i < mins.length; i += 1) {
    const t = start + (i + 0.5) * binSpan;
    const x = timeToX(t);
    if (x < -strokeWidth || x > width + strokeWidth) continue;
    if (visualFirstMode() && !serverBoomDisplayActive(serverSeries, i, tile)) continue;
    const y1 = sampleY(maxs[i], laneTop, laneHeight, tile);
    const y2 = sampleY(mins[i], laneTop, laneHeight, tile);
    ctx.beginPath();
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPeakGhost(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const mins = tile?.mins || [];
  const maxs = tile?.maxs || [];
  if (!mins.length || !maxs.length) return;
  const serverSeries = visualFirstMode() ? serverBoomSeries(tile, Math.max(mins.length, maxs.length)) : null;
  if (visualFirstMode() && !serverSeries) return;
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = span / Math.max(1, mins.length);
  const strokeWidth = Math.max(0.45, Math.min(2, binSpan * pixelsPerSecond()));
  ctx.save();
  ctx.globalAlpha = HYPER_RMS_PEAK_GHOST_ALPHA;
  ctx.strokeStyle = stemStroke(tile);
  ctx.lineWidth = strokeWidth;
  for (let i = 0; i < mins.length; i += 1) {
    const t = start + (i + 0.5) * binSpan;
    const x = timeToX(t);
    if (x < -strokeWidth || x > width + strokeWidth) continue;
    if (visualFirstMode() && !serverBoomDisplayActive(serverSeries, i, tile)) continue;
    const y1 = sampleY(maxs[i], laneTop, laneHeight, tile);
    const y2 = sampleY(mins[i], laneTop, laneHeight, tile);
    ctx.beginPath();
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPeakBoundaryOverlay(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const mins = tile?.mins || [];
  const maxs = tile?.maxs || [];
  if (!mins.length || !maxs.length) return;
  const serverSeries = visualFirstMode() ? serverBoomSeries(tile, Math.max(mins.length, maxs.length)) : null;
  if (visualFirstMode() && !serverSeries) return;
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = span / Math.max(1, mins.length);
  ctx.save();
  ctx.globalAlpha = RMS_INSPECTION_PEAK_ALPHA;
  ctx.strokeStyle = stemStroke(tile);
  ctx.lineWidth = Math.max(0.75, Math.min(1.8, binSpan * pixelsPerSecond()));
  const runs = serverSeries ? serverBoomActiveRuns(serverSeries, Math.min(mins.length, maxs.length), tile) : [
    { startIndex: 0, endIndex: Math.min(mins.length, maxs.length) },
  ];
  runs.forEach((run) => {
    if (run.endIndex - run.startIndex < 2) return;
    ctx.beginPath();
    for (let i = run.startIndex; i < run.endIndex; i += 1) {
      const t = start + (i + 0.5) * binSpan;
      const x = timeToX(t);
      const y = sampleY(maxs[i], laneTop, laneHeight, tile);
      if (i === run.startIndex) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.beginPath();
    for (let i = run.startIndex; i < run.endIndex; i += 1) {
      const t = start + (i + 0.5) * binSpan;
      const x = timeToX(t);
      const y = sampleY(mins[i], laneTop, laneHeight, tile);
      if (i === run.startIndex) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  ctx.restore();
}

function drawSampleInspectionOverlay(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const samples = tile?.samples || [];
  if (samples.length < 2) return false;
  const serverSeries = visualFirstMode() ? serverBoomSeries(tile) : null;
  if (visualFirstMode() && !serverSeries) return false;
  const sr = Number(tile?.sample_rate || sampleRate());
  const startSample = Number(tile.sample_start || tile.start_sample || 0);
  const visibleStart = Math.max(0, Math.floor(viewportStart * sr) - startSample - 2);
  const visibleEnd = Math.min(samples.length, Math.ceil(viewportEnd * sr) - startSample + 2);
  if (visibleEnd - visibleStart < 2) return false;

  const spacing = pixelsPerSecond() / Math.max(sr, 1);
  if (spacing < RMS_INSPECTION_MIN_SAMPLE_SPACING) return false;

  const segments = [];
  let currentSegment = [];
  for (let index = visibleStart; index < visibleEnd; index += 1) {
    const time = (startSample + index) / sr;
    if (visualFirstMode() && !serverBoomInspectionActiveAtTime(tile, time, serverSeries, 1)) {
      if (currentSegment.length >= 2) segments.push(currentSegment);
      currentSegment = [];
      continue;
    }
    const sample = Number(samples[index]);
    currentSegment.push([timeToX(time), sampleY(sample, laneTop, laneHeight, tile), sample]);
  }
  if (currentSegment.length >= 2) segments.push(currentSegment);
  if (!segments.length) return false;

  ctx.save();
  ctx.strokeStyle = "rgba(17,24,32,0.82)";
  ctx.lineWidth = spacing >= 2 ? 1.35 : 0.85;
  segments.forEach((points) => {
    ctx.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  if (spacing >= RMS_INSPECTION_ZERO_CROSSING_SPACING) {
    const mid = laneTop + laneHeight / 2;
    ctx.strokeStyle = "rgba(17,24,32,0.72)";
    ctx.fillStyle = "rgba(17,24,32,0.72)";
    segments.forEach((points) => {
      for (let i = 1; i < points.length; i += 1) {
        const prev = points[i - 1];
        const curr = points[i];
        if ((prev[2] <= 0 && curr[2] > 0) || (prev[2] >= 0 && curr[2] < 0)) {
          const ratio = Math.abs(prev[2]) / Math.max(Math.abs(prev[2]) + Math.abs(curr[2]), 1e-9);
          const x = prev[0] + (curr[0] - prev[0]) * ratio;
          ctx.beginPath();
          ctx.moveTo(x, mid - 8);
          ctx.lineTo(x, mid + 8);
          ctx.stroke();
          if (spacing >= 8) {
            ctx.beginPath();
            ctx.arc(x, mid, 2.2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    });
  }

  ctx.restore();
  return true;
}

function drawRmsEnvelope(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const rawRms = Array.isArray(tile?.rms) ? tile.rms : [];
  if (rawRms.length < 2) return false;
  const serverSeries = visualFirstMode() ? serverBoomSeries(tile, rawRms.length) : null;
  if (visualFirstMode() && !serverSeries) return false;
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = span / Math.max(1, rawRms.length);
  const rms = smoothRmsValues(rawRms, binSpan);
  drawEnergyChunks(ctx, width, laneTop, laneHeight, tile, rawRms, binSpan);
  const runs = serverSeries ? serverBoomActiveRuns(serverSeries, rms.length, tile) : [
    { startIndex: 0, endIndex: rms.length },
  ];

  ctx.fillStyle = stemFill(tile);
  runs.forEach((run) => {
    if (run.endIndex <= run.startIndex) return;
    ctx.beginPath();
    for (let i = run.startIndex; i < run.endIndex; i += 1) {
      const t = start + (i + 0.5) * binSpan;
      const x = timeToX(t);
      const y = rmsY(rms[i], laneTop, laneHeight, tile, 1);
      if (i === run.startIndex) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = run.endIndex - 1; i >= run.startIndex; i -= 1) {
      const t = start + (i + 0.5) * binSpan;
      ctx.lineTo(timeToX(t), rmsY(rms[i], laneTop, laneHeight, tile, -1));
    }
    ctx.closePath();
    ctx.fill();
  });

  ctx.save();
  ctx.strokeStyle = stemStroke(tile);
  ctx.lineWidth = 1.45;
  ctx.globalAlpha = 0.88;
  runs.forEach((run) => {
    if (run.endIndex - run.startIndex < 2) return;
    ctx.beginPath();
    for (let i = run.startIndex; i < run.endIndex; i += 1) {
      const t = start + (i + 0.5) * binSpan;
      const x = timeToX(t);
      const y = rmsY(rms[i], laneTop, laneHeight, tile, 1);
      if (i === run.startIndex) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.beginPath();
    for (let i = run.startIndex; i < run.endIndex; i += 1) {
      const t = start + (i + 0.5) * binSpan;
      const x = timeToX(t);
      const y = rmsY(rms[i], laneTop, laneHeight, tile, -1);
      if (i === run.startIndex) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  ctx.restore();

  drawPeakBoundaryOverlay(ctx, width, laneTop, laneHeight, tile);
  drawPeakGhost(ctx, width, laneTop, laneHeight, tile);
  drawSampleInspectionOverlay(ctx, width, laneTop, laneHeight, tile);
  return true;
}

function drawSamples(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const samples = tile?.samples || [];
  if (samples.length < 2) return;
  const serverSeries = visualFirstMode() ? serverBoomSeries(tile) : null;
  if (visualFirstMode() && !serverSeries) return;
  const sr = Number(tile?.sample_rate || sampleRate());
  const startSample = Number(tile.sample_start || tile.start_sample || 0);
  const visibleStart = Math.max(0, Math.floor(viewportStart * sr) - startSample - 2);
  const visibleEnd = Math.min(samples.length, Math.ceil(viewportEnd * sr) - startSample + 2);
  if (visibleEnd - visibleStart < 2) return;
  const segments = [];
  let currentSegment = [];
  for (let index = visibleStart; index < visibleEnd; index += 1) {
    const time = (startSample + index) / sr;
    if (visualFirstMode() && !serverBoomInspectionActiveAtTime(tile, time, serverSeries, 1)) {
      if (currentSegment.length >= 2) segments.push(currentSegment);
      currentSegment = [];
      continue;
    }
    const sample = Number(samples[index]);
    currentSegment.push([timeToX(time), sampleY(sample, laneTop, laneHeight, tile), sample]);
  }
  if (currentSegment.length >= 2) segments.push(currentSegment);
  if (!segments.length) return;

  ctx.strokeStyle = stemStroke(tile);
  ctx.lineWidth = 1.4;
  segments.forEach((points) => {
    ctx.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  const firstSegment = segments.find((points) => points.length >= 2);
  const spacing = firstSegment ? Math.abs(firstSegment[1][0] - firstSegment[0][0]) : 0;
  if (spacing >= 5) {
    ctx.fillStyle = stemStroke(tile);
    segments.forEach((points) => {
      for (const [x, y] of points) {
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  if (spacing >= 3) {
    ctx.strokeStyle = "#111820";
    ctx.fillStyle = "#111820";
    segments.forEach((points) => {
      for (let i = 1; i < points.length; i += 1) {
        const prev = points[i - 1];
        const curr = points[i];
        if ((prev[2] <= 0 && curr[2] > 0) || (prev[2] >= 0 && curr[2] < 0)) {
          const ratio = Math.abs(prev[2]) / Math.max(Math.abs(prev[2]) + Math.abs(curr[2]), 1e-9);
          const x = prev[0] + (curr[0] - prev[0]) * ratio;
          const mid = laneTop + laneHeight / 2;
          ctx.beginPath();
          ctx.moveTo(x, mid - 7);
          ctx.lineTo(x, mid + 7);
          ctx.stroke();
          if (spacing >= 8) {
            ctx.beginPath();
            ctx.arc(x, mid, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    });
  }
}

function markerAnchorClientX(clientX = null) {
  const rect = $("waveWrapper").getBoundingClientRect();
  if (Number.isFinite(Number(clientX))) {
    return Math.min(Math.max(Number(clientX), rect.left), rect.right);
  }
  return rect.left + rect.width / 2;
}

function pointerMidpointClientX(points = [...activePointers.values()]) {
  if (!points.length) return null;
  return points.reduce((sum, point) => sum + Number(point.x || 0), 0) / points.length;
}

function drawMarker(parent, time, className, label, options = {}) {
  if (!currentItem || !waveDuration || time === null || time === undefined) return;
  const t = Number(time);
  if (t < viewportStart || t > viewportEnd) return;
  const left = timeToX(t);
  const wrapperWidth = $("waveWrapper").clientWidth || 1;
  if (left < -80 || left > wrapperWidth + 80) return;
  const marker = document.createElement("div");
  marker.className = `marker ${className}`;
  if (left > wrapperWidth - 70) marker.classList.add("edgeRight");
  marker.style.left = `${left}px`;
  const span = document.createElement("span");
  span.textContent = label;
  if (options.labelTop !== undefined) span.style.top = `${options.labelTop}px`;
  marker.appendChild(span);
  parent.appendChild(marker);
}

function drawMarkers() {
  const overlay = $("waveOverlay");
  overlay.innerHTML = "";
  if (!currentItem || !waveDuration) return;
  if (visualFirstMode()) {
    const times = markerTimes();
    drawMarker(overlay, times.musicalOne, "gridOne", "ONE", { labelTop: 4 });
    drawMarker(overlay, times.visual, "visualEdge", "VISUAL", { labelTop: 24 });
    drawMarker(overlay, times.zero, "zeroDiagnostic", "ZERO", { labelTop: 66 });
    drawMarker(overlay, times.crest, "bodyCrest", "BODY CREST", { labelTop: 87 });
    if (viewportDuration() <= 1.0) {
      times.alternates.forEach((time, index) => {
        drawMarker(overlay, time, "alternateAttack", `A${index + 1}`, { labelTop: 108 + index * 20 });
      });
    }
  }
  const blue = blueMarkerInfo();
  drawMarker(overlay, blue.time, "grid", visualFirstMode() && blue.source === "als_impact" ? "BLUE ALS" : "BLUE", {
    labelTop: visualFirstMode() ? 45 : 24,
  });
  if (!visualFirstMode()) drawMarker(overlay, kneeMarkerTime(), "knee", "KNEE");
  const showCandidateMarkers = !visualFirstMode();
  if (showCandidateMarkers) {
    (currentItem.top_10_candidates || []).forEach((cand, index) => {
      const labelTop = isMobileLayout() ? 4 + Math.min(index, 9) * 19 : 4;
      drawMarker(overlay, candidateTime(cand), "candidate", `#${cand.rank}`, { labelTop });
    });
  }
  if (userPick !== null) drawMarker(overlay, userPick, "user", "YOU");
  if (userPick === null && correctionMode && hoverPlaceTime !== null) {
    drawMarker(overlay, hoverPlaceTime, "placementPreview", "1.1.1");
  }
  const playTime = playbackOriginalTime();
  if (playTime !== null) drawMarker(overlay, playTime, "playhead", "PLAY");
}

function drawWaveform() {
  const { ctx, width, height } = setupCanvas();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const lanes = waveLaneSources();
  const laneCount = Math.max(1, lanes.length);
  const laneHeight = height / laneCount;
  lanes.forEach((lane, index) => {
    const top = index * laneHeight;
    drawLaneFrame(ctx, width, top, laneHeight, lane, index);
  });
  drawGrid(ctx, width, height);
  lanes.forEach((lane, index) => {
    if (lane?.ok === false) return;
    const top = index * laneHeight;
    if (waveformVisualMode === "boom" && drawBoomBodyBars(ctx, width, top, laneHeight, lane)) return;
    if (waveformVisualMode === "rms" && drawRmsEnvelope(ctx, width, top, laneHeight, lane)) return;
    if (lane?.mode === "samples") drawSamples(ctx, width, top, laneHeight, lane);
    else drawPeaks(ctx, width, top, laneHeight, lane);
  });
  drawBpmClock(ctx, width, height);
  drawPhraseGuides(ctx, width, height);
  lanes.forEach((lane, index) => drawLaneLabel(ctx, index * laneHeight, lane, index));
  renderStemControls();
  drawMarkers();
}

function scheduleDrawWaveform() {
  if (drawFrameId !== null) return;
  drawFrameId = window.requestAnimationFrame(() => {
    drawFrameId = null;
    drawWaveform();
  });
}

function renderCandidates(item) {
  const list = $("candidateList");
  list.innerHTML = "";
  if (visualFirstMode()) return;
  for (const cand of item.top_10_candidates || []) {
    const li = document.createElement("li");
    const dp = drumMetric(cand, "drumprint_pattern_score");
    const stability = drumMetric(cand, "post_drop_pattern_stability");
    const fake = drumMetric(cand, "fake_hit_penalty");
    const later = drumMetric(cand, "later_drop_match_score");
    const groove = drumMetric(cand, "sustained_full_groove_score");
    const immediateGroove = drumMetric(cand, "immediate_groove_start_score");
    const grooveStable = drumMetric(cand, "groove_stability");
    const contrast = drumMetric(cand, "pre_drop_contrast");
    const micro = cand.microalign || {};
    const microConf = Number(micro.micro_confidence ?? cand.micro_confidence ?? 0);
    const snapMs = Number(micro.snap_offset_ms ?? cand.snap_offset_ms ?? 0);
    const score = Number(cand.confidence_score || cand.score || 0);
    const rankBits = [];
    if (cand.region_model_rank) rankBits.push(`R${cand.region_model_rank}`);
    if (cand.handcrafted_rank) rankBits.push(`H${cand.handcrafted_rank}`);
    if (cand.model_rank) rankBits.push(`M${cand.model_rank}`);
    const t = candidateTime(cand);
    const barPriorText = formatBarPrior(barPriorForCandidate(cand));
    const clockText = formatBpmClock(bpmClockForCandidate(cand));
    const isPicked = pickedCandidate?.picked_candidate_rank === cand.rank;
    const isBlueAnchor = visualFirstMode() && blueAnchorCandidate?.rank === cand.rank;
    const isBlueNearest = visualFirstMode() && blueMarkerInfo().candidate?.rank === cand.rank;
    li.className = visualFirstMode()
      ? (isPicked || isBlueAnchor || isBlueNearest ? "selected" : "")
      : (cand.selected || isPicked ? "selected" : "");
    li.innerHTML = `
      <button class="candidateButton" type="button" aria-label="Pick candidate ${cand.rank}">
        <span class="candidateTop">
          <span class="candidateRank">${cand.rank}</span>
          <span class="candidateTime">${fmtTime(t)}</span>
        </span>
        <span class="candidateBadges">
          <span>score ${score.toFixed(3)}</span>
          <span>micro ${microConf.toFixed(3)}</span>
          ${barPriorText ? `<span>${barPriorText}</span>` : ""}
          ${clockText ? `<span>${clockText}</span>` : ""}
          <span>${rankBits.join(" ") || "ranked"}</span>
        </span>
        <span class="candidateDetail">${cand.reason || "candidate marker"}</span>
        <span class="candidateDetail">snap ${snapMs.toFixed(1)}ms | groove ${groove.toFixed(3)} | drum ${dp.toFixed(3)} | fake ${fake.toFixed(3)}</span>
      </button>
    `;
    li.querySelector(".candidateButton").addEventListener("click", () => {
      if (t === null) {
        setStatus(`Candidate #${cand.rank} has no marker time.`);
        return;
      }
      if (visualFirstMode()) {
        setBlueAnchorCandidate(cand);
        const focusTime = Number(t);
        setViewportAround(focusTime, Math.min(visualCandidateZoomRadius(), VISUAL_DIRECT_PLACE_VIEW_SECONDS / 2), "custom");
        setCorrectionMode(false);
        renderCandidates(currentItem);
        setStatus(`Candidate #${cand.rank} focused at ${fmtTime(focusTime)}. Click waveform to play/seek. Press PLACE 1.1.1 when you are ready to mark.`);
        return;
      }
      setUserPick(Number(t), cand);
      setCorrectionMode(false);
      renderCandidates(currentItem);
      setStatus(`Candidate #${cand.rank} selected. Tap SAVE to write ${fmtTime(userPick)}.`);
    });
    list.appendChild(li);
  }
}

function renderStructureSummary() {
  const grid = structureBeatgrid();
  const first = structureCandidate("first_drop");
  const second = structureCandidate("second_drop");
  const gridTarget = $("structureBeatgrid");
  const firstTarget = $("structureFirstDrop");
  const secondTarget = $("structureSecondDrop");
  if (gridTarget) {
    if (grid) {
      gridTarget.textContent = `grid anchor ${fmtTime(Number(grid.bar_zero_sec || 0))} | conf ${Number(grid.downbeat_confidence || 0).toFixed(2)}`;
    } else {
      gridTarget.textContent = "--";
    }
  }
  if (firstTarget) {
    firstTarget.textContent = first ? formatStructureMarker(first) : "--";
  }
  if (secondTarget) {
    secondTarget.textContent = second ? formatStructureMarker(second) : "--";
  }
}

function destroyWave() {
  stopStemPlayers();
  if (audioPlayer) {
    audioPlayer.pause();
    audioPlayer = null;
  }
  audioClipStartSec = 0;
  audioClipEndSec = 0;
  pendingPlaybackTime = null;
  playbackMarkerTime = null;
  if (playbackSeekTimer) {
    window.clearTimeout(playbackSeekTimer);
    playbackSeekTimer = null;
  }
  stopPlaybackFrame();
  updatePlaybackControls();
  setPlaybackStatus("");
  if (tileRequestTimer) {
    window.clearTimeout(tileRequestTimer);
    tileRequestTimer = null;
  }
  if (tileAbortController) {
    tileAbortController.abort();
    tileAbortController = null;
  }
  if (drawFrameId !== null) {
    window.cancelAnimationFrame(drawFrameId);
    drawFrameId = null;
  }
  tileRequestSeq += 1;
  $("waveOverlay").innerHTML = "";
  $("waveControlsOverlay").innerHTML = "";
  waveTile = null;
  waveTiles = [];
  waveDuration = 0;
  viewportStart = 0;
  viewportEnd = 1;
  syncZoomControls();
  drawWaveform();
}

function syncViewButtons() {
  $("fullTrackBtn").classList.toggle("active", waveView === "full");
  $("aiWindowBtn").classList.toggle("active", waveView === "window");
}

function syncZoomControls() {
  const duration = viewportDuration();
  const pps = pixelsPerSecond();
  const msPerPixel = 1000 / Math.max(pps, 0.001);
  $("zoomSlider").max = "1000";
  if (waveDuration > 0) {
    const full = Math.max(waveDuration, minViewSeconds());
    const ratio = Math.max(0, Math.min(1, 1 - Math.log(duration / minViewSeconds()) / Math.log(full / minViewSeconds())));
    $("zoomSlider").value = String(Math.round(ratio * 1000));
  } else {
    $("zoomSlider").value = "0";
  }
  $("zoomLabel").textContent = `${duration < 1 ? (duration * 1000).toFixed(2) + " ms" : fmtTime(duration)} | ${msPerPixel.toFixed(4)} ms/px`;
  updateAudioStatus();
  updateGridMarkerUi();
}

function updateAudioStatus() {
  if (!currentItem || !waveDuration) return;
  const primary = waveTile || usableWaveTiles()[0] || null;
  const stemCount = Math.max(1, usableWaveTiles().length || availableStemCount());
  const usesRmsBins = (waveformVisualMode === "rms" || waveformVisualMode === "boom") && primary?.rms?.length;
  const mode = usesRmsBins
    ? waveformModeText()
    : primary?.mode === "samples" ? "raw samples" : "ultra min/max peaks";
  const cache = primary?.cache_hit ? "cache hit" : "cache build";
  const detailCount =
    usesRmsBins
      ? `${(primary?.rms || []).length} rms bins`
      : primary?.mode === "samples"
      ? `${(primary?.samples || []).length} samples`
      : `${(primary?.mins || []).length} bins`;
  const secPerBin =
    usesRmsBins
      ? Math.max(0, Number(primary?.end_sec || 0) - Number(primary?.start_sec || 0)) / Math.max((primary?.rms || []).length, 1)
      : primary?.mode === "samples"
      ? 1 / Math.max(sampleRate(), 1)
      : Math.max(0, Number(primary?.end_sec || 0) - Number(primary?.start_sec || 0)) / Math.max((primary?.mins || []).length, 1);
  const maximizerText = usesRmsBins
    ? ` | max ${rmsMaximizerMakeup(primary).toFixed(2)}x`
    : "";
  const inspectionText = usesRmsBins
    ? ` | ${sampleInspectionLabel(primary)}`
    : "";
  $("audioStatus").textContent =
    `${stemCount} stem ${stemCount === 1 ? "lane" : "lanes"} | ${mode}${maximizerText}${inspectionText} | ${detailCount} | ${(secPerBin * 1000).toFixed(3)} ms/bin | ${cache} | ` +
    `${fmtTime(viewportStart)} to ${fmtTime(viewportEnd)} | sr ${sampleRate()} Hz`;
}

function setViewport(start, end, nextView = waveView) {
  if (!currentItem || !waveDuration) return;
  const minSpan = minViewSeconds();
  let nextStart = Number(start);
  let nextEnd = Number(end);
  if (!Number.isFinite(nextStart) || !Number.isFinite(nextEnd)) return;
  if (nextEnd < nextStart) [nextStart, nextEnd] = [nextEnd, nextStart];
  if (nextEnd - nextStart < minSpan) {
    const center = (nextStart + nextEnd) / 2;
    nextStart = center - minSpan / 2;
    nextEnd = center + minSpan / 2;
  }
  const span = nextEnd - nextStart;
  if (nextStart < 0) {
    nextEnd = Math.min(waveDuration, span);
    nextStart = 0;
  }
  if (nextEnd > waveDuration) {
    nextStart = Math.max(0, waveDuration - span);
    nextEnd = waveDuration;
  }
  viewportStart = snapTimeToSample(nextStart);
  viewportEnd = snapTimeToSample(Math.max(viewportStart + minSpan, nextEnd));
  waveView = nextView;
  syncViewButtons();
  syncZoomControls();
  if (!visualDirectPlaceReady()) hoverPlaceTime = null;
  $("waveWrapper")?.classList.toggle("directPlace", visualDirectPlaceReady() && correctionMode);
  scheduleDrawWaveform();
  updatePlaceButton();
  scheduleWaveformTile();
}

function scheduleWaveformTile(delay = TILE_REQUEST_DEBOUNCE_MS) {
  if (!currentItem || !waveDuration) return;
  if (tileAbortController) {
    tileAbortController.abort();
    tileAbortController = null;
  }
  if (tileRequestTimer) window.clearTimeout(tileRequestTimer);
  tileRequestTimer = window.setTimeout(() => {
    tileRequestTimer = null;
    requestWaveformTile();
  }, Math.max(0, Number(delay) || 0));
}

function requestWaveformTile() {
  if (!currentItem || !waveDuration) return;
  const seq = ++tileRequestSeq;
  if (tileAbortController) tileAbortController.abort();
  tileAbortController = new AbortController();
  const controller = tileAbortController;
  const canvas = $("waveCanvas");
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const renderWidth = Math.max(400, Math.round((canvas.clientWidth || $("waveWrapper").clientWidth || 1000) * dpr));
  const width = Math.min(WAVEFORM_MAX_TILE_BINS, Math.max(renderWidth, Math.round(renderWidth * WAVEFORM_DETAIL_MULTIPLIER)));
  const params = new URLSearchParams({
    id: currentItem.id,
    start: viewportStart.toFixed(9),
    end: viewportEnd.toFixed(9),
    width: String(width),
  });
  if (!waveTile) $("audioStatus").textContent = "Loading waveform tile...";
  fetchJson(`/api/waveform_stems_tile?${params.toString()}`, { signal: controller.signal })
    .then((data) => {
      if (seq !== tileRequestSeq) return;
      if (tileAbortController === controller) tileAbortController = null;
      waveTiles = Array.isArray(data.tiles) ? data.tiles : [data];
      waveTile =
        waveTiles.find((tile) => tile && tile.ok !== false && tile.role === data.primary_role) ||
        waveTiles.find((tile) => tile && tile.ok !== false) ||
        null;
      waveDuration = Number(data.duration || waveTile?.duration || waveDuration || currentItem.full_duration_sec || 0);
      drawWaveform();
      syncZoomControls();
      updatePlaceButton();
    })
    .catch((err) => {
      if (err.name === "AbortError") return;
      if (tileAbortController === controller) tileAbortController = null;
      if (seq === tileRequestSeq) $("audioStatus").textContent = `Waveform tile error: ${err.message}`;
    });
}

function setViewportAround(center, radius, nextView = "preset") {
  const c = clampOriginalTime(center);
  const r = Math.max(minViewSeconds() / 2, Number(radius));
  setViewport(c - r, c + r, nextView);
}

function focusAiWindow() {
  if (!currentItem || !waveDuration) return;
  const ai = optionalMarkerTime(currentItem.ai_pick);
  if (ai === null) return resetZoom();
  setViewport(ai - DEFAULT_WINDOW_BEFORE, ai + DEFAULT_WINDOW_AFTER, "window");
}

function resetZoom() {
  if (!waveDuration) return;
  setViewport(0, waveDuration, "full");
}

function setWaveView(nextView) {
  if (!currentItem) return;
  if (nextView === "window") focusAiWindow();
  else resetZoom();
}

function setZoomFactor(factor, anchorClientX = null) {
  if (!currentItem || !waveDuration) return;
  const oldSpan = viewportDuration();
  const newSpan = Math.max(minViewSeconds(), Math.min(waveDuration, oldSpan * Number(factor)));
  setViewportWithAnchor(newSpan, timeFromClientX(markerAnchorClientX(anchorClientX)), anchorClientX);
}

function setViewportWithAnchor(span, anchorTime, anchorClientX = null) {
  if (!currentItem || !waveDuration) return;
  const rect = $("waveWrapper").getBoundingClientRect();
  const anchor = markerAnchorClientX(anchorClientX);
  const anchorX = Math.min(Math.max(anchor - rect.left, 0), rect.width || 1);
  const ratio = (rect.width || 1) > 0 ? anchorX / (rect.width || 1) : 0.5;
  const nextSpan = Math.max(minViewSeconds(), Math.min(waveDuration, Number(span)));
  const nextStart = Number(anchorTime) - nextSpan * ratio;
  setViewport(nextStart, nextStart + nextSpan, "custom");
}

function zoomBy(delta, anchorClientX = null) {
  setZoomFactor(Number(delta) > 0 ? 0.4 : 2.5, anchorClientX);
}

function setZoomFromSlider(value) {
  if (!waveDuration) return;
  const ratio = Math.max(0, Math.min(1, Number(value) / 1000));
  const minSpan = minViewSeconds();
  const full = Math.max(waveDuration, minSpan);
  const span = minSpan * Math.exp((1 - ratio) * Math.log(full / minSpan));
  const center = (viewportStart + viewportEnd) / 2;
  setViewport(center - span / 2, center + span / 2, "custom");
}

function panBy(pixels) {
  if (!waveDuration) return;
  const shift = Number(pixels) / Math.max(pixelsPerSecond(), 0.001);
  setViewport(viewportStart + shift, viewportEnd + shift, waveView);
}

function normalizedWheelDelta(value, event) {
  let out = Number(value || 0);
  if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) out *= 16;
  if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) out *= Math.max(window.innerHeight || 800, 1);
  return Math.max(-600, Math.min(600, out));
}

function acceleratedZoomFactor(factor) {
  const safe = Math.max(0.05, Math.min(20, Number(factor) || 1));
  return Math.pow(safe, PINCH_ZOOM_ACCELERATION);
}

function scrollToOriginalTime(originalTime) {
  if (!waveDuration) return;
  const span = viewportDuration();
  const center = clampOriginalTime(originalTime);
  setViewport(center - span / 2, center + span / 2, waveView);
}

function timeFromClientX(clientX) {
  const rect = $("waveWrapper").getBoundingClientRect();
  const x = Math.min(Math.max(Number(clientX) - rect.left, 0), rect.width || 1);
  return clampOriginalTime(viewportStart + x / Math.max(pixelsPerSecond(), 0.001));
}

async function placeAtClientX(clientX) {
  if (!correctionMode || !currentItem || !waveDuration || placeValidationInFlight) return;
  const target = timeFromClientX(clientX);
  const placement = resolveBoomPlacementTime(target);
  if (!placement.ok) {
    const evidence = placement.evidence || {};
    const postPct = Math.round((Number(evidence.normalizedPost) || 0) * 100);
    const peakPct = Math.round((Number(evidence.normalizedPeak) || 0) * 100);
    const nearest = placement.snap?.time;
    const nearestText = nearest !== undefined && nearest !== null ? ` Nearest Boom edge is ${fmtTime(nearest)}.` : "";
    setStatus(
      `Rejected ${fmtTime(target)}: too low/flat for a drop front (${evidence.reason}, body ${postPct}%, peak ${peakPct}%). ` +
        `Zoom to the first dark boom body edge before placing 1.1.1.${nearestText}`,
    );
    setPlaybackStatus(`Low/flat point rejected at ${fmtTime(target)}`);
    return;
  }
  placeValidationInFlight = true;
  updatePlaceButton();
  setStatus(`Checking Boom proof at ${fmtTime(placement.time)} before placing 1.1.1...`);
  try {
    const serverValidation = await validateVisualPlacementOnServer(placement.time);
    if (!serverValidation.valid) {
      setStatus(`Rejected ${fmtTime(placement.time)}: ${serverValidation.error || "not a validated Boom front edge"}`);
      setPlaybackStatus(`Server Boom proof rejected ${fmtTime(placement.time)}`);
      return;
    }
    setUserPick(placement.time);
    hoverPlaceTime = null;
    setCorrectionMode(false);
    const snapText = placement.snapped ? ` Snapped from ${fmtTime(target)} to the Boom front edge.` : "";
    setStatus(`1.1.1 placed at ${fmtTime(userPick)} (${userPick.toFixed(6)}s).${snapText} Tap SAVE PLACED when it looks right.`);
  } catch (err) {
    setStatus(`Placement check failed: ${err.message}`);
  } finally {
    placeValidationInFlight = false;
    updatePlaceButton();
  }
}

function visualZoomAtClientX(clientX) {
  if (!visualFirstMode() || correctionMode || !currentItem || !waveDuration) return;
  const center = timeFromClientX(clientX);
  const span = viewportDuration();
  let radius = 5;
  if (span > 90) radius = 8;
  else if (span > 24) radius = 4;
  else if (span > 8) radius = 1;
  else if (span > 2) radius = 0.25;
  else if (span > 0.4) radius = 0.05;
  else if (span > 0.08) radius = 0.008;
  else radius = 0.002;
  setViewportAround(center, radius, "custom");
  const nextHint = visualDirectPlaceReady() ? "Click waveform to play/seek, or press PLACE 1.1.1 to mark." : "Click again to zoom closer.";
  setPlaybackStatus(`Zoomed to ${fmtTime(center)}. ${nextHint}`);
}

function loadWaveform(item) {
  destroyWave();
  waveDuration = Number(item.full_duration_sec || item.duration_sec || 0);
  if (!waveDuration) waveDuration = Math.max(Number(item.ai_pick || 0) + 30, 60);
  const previewStart = Number(item.preview_offset_sec || 0);
  const previewEnd = previewStart + Number(item.preview_duration_sec || 0);
  audioClipStartSec = previewStart;
  audioClipEndSec = previewEnd;
  updatePlaybackControls();
  waveView = visualFirstMode() ? "full" : "window";
  syncViewButtons();
  setupCanvas();
  if (visualFirstMode()) resetZoom();
  else focusAiWindow();
}

function activeCenterTime() {
  if (userPick !== null) return userPick;
  if (selectedMicroTime() !== null) return selectedMicroTime();
  return optionalMarkerTime(currentItem?.ai_pick) ?? waveDuration / 2;
}

function applyPreset(name) {
  if (!currentItem || !waveDuration) return;
  if (name === "full") return resetZoom();
  if (name === "window") return focusAiWindow();
  const radii = {
    "5s": 5,
    "500ms": 0.5,
    "50ms": 0.05,
    "5ms": 0.005,
    "1ms": 0.001,
  };
  setViewportAround(activeCenterTime(), radii[name] || 5, "preset");
}

function playbackOriginalTime() {
  if (!audioPlayer) return playbackMarkerTime;
  if (pendingPlaybackTime !== null) return pendingPlaybackTime;
  if (playbackMarkerTime !== null && audioPlayer.paused && Number(audioPlayer.currentTime || 0) <= 0.001) {
    return playbackMarkerTime;
  }
  playbackMarkerTime = audioClipStartSec + Number(audioPlayer.currentTime || 0);
  return playbackMarkerTime;
}

function playbackClipEnd() {
  if (audioClipEndSec > audioClipStartSec) return audioClipEndSec;
  if (audioPlayer && Number.isFinite(audioPlayer.duration)) return audioClipStartSec + Number(audioPlayer.duration || 0);
  return audioClipStartSec;
}

function updatePlaybackControls() {
  const seek = $("playSeek");
  const time = $("playTime");
  const clip = $("playClip");
  const wavePlayButton = $("wavePlayBtn");
  if (wavePlayButton) {
    const isPlaying = Boolean(audioPlayer && !audioPlayer.paused);
    wavePlayButton.textContent = isPlaying ? "PAUSE" : "PLAY";
    wavePlayButton.classList.toggle("playing", isPlaying);
    wavePlayButton.disabled = !currentItem;
  }
  if (!seek || !time || !clip) return;
  if (!audioPlayer && !currentItem) {
    seek.disabled = true;
    seek.min = "0";
    seek.max = "1";
    seek.value = "0";
    time.textContent = "--";
    clip.textContent = "no clip loaded";
    drawMarkers();
    return;
  }

  const start = audioClipStartSec;
  const end = Math.max(start + 0.001, playbackClipEnd());
  const current = clampOriginalTime(playbackOriginalTime() ?? start);
  seek.disabled = !currentItem || !waveDuration;
  seek.min = "0";
  seek.max = Math.max(waveDuration || Number(currentItem?.full_duration_sec || 0) || end, 0.001).toFixed(3);
  seek.step = "0.001";
  seek.value = current.toFixed(3);
  time.textContent = fmtTime(current);
  clip.textContent = audioPlayer ? `clip ${fmtTime(start)}-${fmtTime(end)}` : "no clip loaded";
  drawMarkers();
}

function playbackFrame() {
  playbackFrameId = null;
  updatePlaybackControls();
  if (audioPlayer && !audioPlayer.paused) {
    playbackFrameId = window.requestAnimationFrame(playbackFrame);
  }
}

function startPlaybackFrame() {
  if (playbackFrameId === null) playbackFrameId = window.requestAnimationFrame(playbackFrame);
}

function stopPlaybackFrame() {
  if (playbackFrameId !== null) {
    window.cancelAnimationFrame(playbackFrameId);
    playbackFrameId = null;
  }
}

function stopStemPlayer(role) {
  const player = stemAudioPlayers[role];
  if (!player) return;
  try {
    player.pause();
    player.src = "";
    player.load();
  } catch (_) {
    // Best-effort cleanup for secondary preview audio.
  }
  delete stemAudioPlayers[role];
}

function stopStemPlayers() {
  Object.keys(stemAudioPlayers).forEach(stopStemPlayer);
}

function pauseStemPlayers() {
  Object.values(stemAudioPlayers).forEach((player) => {
    try {
      player.pause();
    } catch (_) {
      // Ignore secondary audio pause failures.
    }
  });
}

function stemAudioRangeUrl(role, start, end) {
  const safeStart = clampOriginalTime(start);
  const safeEnd = clampOriginalTime(Math.max(safeStart + 0.05, Number(end)));
  const params = new URLSearchParams({
    id: currentItem.id,
    role: String(role || "drums"),
    start: safeStart.toFixed(6),
    end: safeEnd.toFixed(6),
  });
  return { url: `/media/audio?${params.toString()}`, start: safeStart, end: safeEnd };
}

function seekStemPlayersTo(originalTime) {
  const target = Math.max(0, clampOriginalTime(originalTime) - audioClipStartSec);
  Object.values(stemAudioPlayers).forEach((player) => {
    const apply = () => {
      try {
        player.currentTime = target;
      } catch (_) {
        // Some browsers reject seeks before metadata is ready.
      }
    };
    if (player.readyState >= 1) apply();
    else player.addEventListener("loadedmetadata", apply, { once: true });
  });
}

function startStemPlayer(role, shouldPlay = true) {
  if (!currentItem || !audioPlayer || isStemMuted(role) || !stemSourceForRole(role)) return;
  if (!shouldPlay) return;
  const range = stemAudioRangeUrl(role, audioClipStartSec, playbackClipEnd());
  let player = stemAudioPlayers[role];
  if (!player || player.src !== new URL(range.url, window.location.href).href) {
    if (player) stopStemPlayer(role);
    player = new Audio();
    player.preload = "auto";
    player.addEventListener("error", () => setPlaybackStatus(`${role} preview unavailable`));
    player.src = range.url;
    player.load();
    stemAudioPlayers[role] = player;
  }
  const target = Math.max(0, Number(audioPlayer.currentTime || 0));
  const align = () => {
    try {
      if (Math.abs(Number(player.currentTime || 0) - target) > 0.060) player.currentTime = target;
    } catch (_) {
      // Secondary player may not be seekable until metadata is ready.
    }
  };
  if (player.readyState >= 1) align();
  else player.addEventListener("loadedmetadata", align, { once: true });
  if (shouldPlay) player.play().catch(() => setPlaybackStatus(`${role} preview waiting`));
}

function startStemPlayers(shouldPlay = true) {
  PLAYBACK_STEM_ROLES.forEach((role) => {
    if (isStemMuted(role)) stopStemPlayer(role);
    else startStemPlayer(role, shouldPlay);
  });
}

function seekLoadedPlayback(originalTime) {
  if (!audioPlayer) return false;
  const target = clampOriginalTime(originalTime);
  const start = audioClipStartSec;
  const end = playbackClipEnd();
  if (target < start || target > end) return false;
  setAudioPlayerOriginalTime(target);
  updatePlaybackControls();
  return true;
}

function setAudioPlayerOriginalTime(originalTime) {
  if (!audioPlayer) return;
  const originalTarget = clampOriginalTime(originalTime);
  const target = Math.max(0, originalTarget - audioClipStartSec);
  pendingPlaybackTime = originalTarget;
  playbackMarkerTime = originalTarget;
  updatePlaybackControls();
  const applySeek = () => {
    try {
      audioPlayer.currentTime = target;
      pendingPlaybackTime = null;
      seekStemPlayersTo(originalTarget);
      updatePlaybackControls();
    } catch (err) {
      setStatus(`Audio seek failed: ${err.message}`);
    }
  };
  if (audioPlayer.readyState >= 1) applySeek();
  else {
    audioPlayer.addEventListener("loadedmetadata", applySeek, { once: true });
    audioPlayer.load();
  }
}

function prepareAudioPlayer(url, clipStartSec = 0, label = "preview", showReady = true, clipEndSec = null) {
  stopStemPlayers();
  if (audioPlayer) {
    audioPlayer.pause();
    audioPlayer = null;
  }
  audioClipStartSec = Number(clipStartSec || 0);
  audioClipEndSec = Number(clipEndSec || 0);
  pendingPlaybackTime = null;
  playbackMarkerTime = audioClipStartSec;
  stopPlaybackFrame();
  if (!url) {
    if (showReady) setPlaybackStatus("");
    updatePlaybackControls();
    return null;
  }
  audioPlayer = new Audio();
  audioPlayer.preload = "metadata";
  audioPlayer.addEventListener("loadedmetadata", () => {
    if (!(audioClipEndSec > audioClipStartSec) && Number.isFinite(audioPlayer.duration)) {
      audioClipEndSec = audioClipStartSec + Number(audioPlayer.duration || 0);
    }
    updatePlaybackControls();
  });
  audioPlayer.addEventListener("play", () => {
    startPlaybackFrame();
    startStemPlayers(true);
  });
  audioPlayer.addEventListener("pause", () => {
    stopPlaybackFrame();
    pauseStemPlayers();
    updatePlaybackControls();
  });
  audioPlayer.addEventListener("timeupdate", updatePlaybackControls);
  audioPlayer.addEventListener("ended", () => {
    stopPlaybackFrame();
    stopStemPlayers();
    updatePlaybackControls();
    setPlaybackStatus(`${label} ended`);
  });
  audioPlayer.addEventListener("error", () => {
    stopPlaybackFrame();
    stopStemPlayers();
    setPlaybackStatus(`${label} unavailable`);
    setStatus("Audio preview failed. ffmpeg must be installed and the source file must still be reachable.");
  });
  if (showReady) setPlaybackStatus(`Loading ${label}...`);
  audioPlayer.src = url;
  audioPlayer.load();
  updatePlaybackControls();
  return audioPlayer;
}

function playAudioPreview(url, clipStartSec, label, clipEndSec = null, seekToOriginalTime = null) {
  const visualTarget = seekToOriginalTime === null ? null : clampOriginalTime(seekToOriginalTime);
  if (visualTarget !== null) {
    playbackMarkerTime = visualTarget;
    drawMarkers();
  }
  const player = prepareAudioPlayer(url, clipStartSec, label, true, clipEndSec);
  if (!player) {
    setStatus("No short preview audio available. ffmpeg is required so the browser does not load the full track.");
    return;
  }
  if (visualTarget !== null) {
    playbackMarkerTime = visualTarget;
    setAudioPlayerOriginalTime(visualTarget);
    drawMarkers();
  }
  player
    .play()
    .then(() => {
      startStemPlayers(true);
      setPlaybackStatus(`Playing ${label} from ${fmtTime(playbackOriginalTime())}`);
    })
    .catch((err) => setStatus(`Audio play failed: ${err.message}`));
}

function audioRangeUrl(start, end) {
  const safeStart = clampOriginalTime(start);
  const safeEnd = clampOriginalTime(Math.max(safeStart + 0.05, Number(end)));
  const params = new URLSearchParams({
    id: currentItem.id,
    start: safeStart.toFixed(6),
    end: safeEnd.toFixed(6),
  });
  return { url: `/media/audio?${params.toString()}`, start: safeStart, end: safeEnd };
}

function cappedRangeAround(start, end) {
  let safeStart = clampOriginalTime(start);
  let safeEnd = clampOriginalTime(Math.max(safeStart + 0.05, end));
  if (safeEnd - safeStart > MAX_AUDIO_PREVIEW_SECONDS) {
    const center = (safeStart + safeEnd) / 2;
    safeStart = clampOriginalTime(center - MAX_AUDIO_PREVIEW_SECONDS / 2);
    safeEnd = clampOriginalTime(safeStart + MAX_AUDIO_PREVIEW_SECONDS);
  }
  return { start: safeStart, end: safeEnd };
}

function playCurrentZoom() {
  if (!currentItem || !waveDuration) return;
  const range = cappedRangeAround(viewportStart, viewportEnd);
  const preview = audioRangeUrl(range.start, range.end);
  playAudioPreview(preview.url, preview.start, `zoom ${fmtTime(preview.start)}-${fmtTime(preview.end)}`, preview.end);
}

function playAroundMarker() {
  if (!currentItem || !waveDuration) return;
  const center = activeCenterTime();
  const range = cappedRangeAround(center - MARKER_PLAY_BEFORE_SECONDS, center + MARKER_PLAY_AFTER_SECONDS);
  const preview = audioRangeUrl(range.start, range.end);
  playAudioPreview(preview.url, preview.start, `marker ${fmtTime(center)}`, preview.end, center);
  playbackMarkerTime = center;
  drawMarkers();
}

function loadAroundPlaybackTime(originalTime, shouldPlay) {
  if (!currentItem || !waveDuration) return;
  const target = clampOriginalTime(originalTime);
  const range = cappedRangeAround(target - SEEK_PLAY_BEFORE_SECONDS, target + SEEK_PLAY_AFTER_SECONDS);
  const preview = audioRangeUrl(range.start, range.end);
  if (shouldPlay) {
    playAudioPreview(preview.url, preview.start, `seek ${fmtTime(target)}`, preview.end, target);
    return;
  }
  const player = prepareAudioPlayer(preview.url, preview.start, `seek ${fmtTime(target)}`, true, preview.end);
  if (player) {
    setAudioPlayerOriginalTime(target);
    setPlaybackStatus(`Ready at ${fmtTime(target)}`);
  }
}

function loadPlaybackStartingAt(originalTime, shouldPlay) {
  if (!currentItem || !waveDuration) return;
  const target = clampOriginalTime(originalTime);
  const preview = audioRangeUrl(target, target + CLICK_SEEK_AFTER_SECONDS);
  const label = `seek ${fmtTime(target)}`;
  const player = prepareAudioPlayer(preview.url, preview.start, label, true, preview.end);
  if (!player) return;
  setPlaybackStatus(`${shouldPlay ? "Playing" : "Ready at"} ${fmtTime(target)}`);
  if (shouldPlay) {
    player
      .play()
      .then(() => setPlaybackStatus(`Playing from ${fmtTime(target)}`))
      .catch((err) => setStatus(`Audio play failed: ${err.message}`));
  }
}

function seekPlaybackToOriginalTime(originalTime) {
  const wasPlaying = Boolean(audioPlayer && !audioPlayer.paused);
  if (seekLoadedPlayback(originalTime)) {
    if (wasPlaying) audioPlayer.play().catch((err) => setStatus(`Audio play failed: ${err.message}`));
    else setPlaybackStatus(`Ready at ${fmtTime(playbackOriginalTime())}`);
    return;
  }
  loadPlaybackStartingAt(originalTime, wasPlaying);
}

function seekPlaybackToClientX(clientX) {
  if (!currentItem || !waveDuration) return;
  const target = timeFromClientX(clientX);
  setPlaybackStatus(`Seeking to ${fmtTime(target)}`);
  loadPlaybackStartingAt(target, Boolean(audioPlayer && !audioPlayer.paused));
}

function playPlaybackFromClientX(clientX) {
  if (!currentItem || !waveDuration) return;
  const target = timeFromClientX(clientX);
  setPlaybackStatus(`Playing from ${fmtTime(target)}`);
  loadPlaybackStartingAt(target, true);
}

function previewPlaybackSeek(originalTime) {
  pendingPlaybackTime = clampOriginalTime(originalTime);
  updatePlaybackControls();
  setPlaybackStatus(`Cue ${fmtTime(pendingPlaybackTime)}`);
}

function commitPlaybackSeek(originalTime) {
  if (playbackSeekTimer) {
    window.clearTimeout(playbackSeekTimer);
    playbackSeekTimer = null;
  }
  const target = clampOriginalTime(originalTime);
  setPlaybackStatus(`Seeking to ${fmtTime(target)}`);
  seekPlaybackToOriginalTime(target);
}

function schedulePlaybackSeek(originalTime) {
  const target = clampOriginalTime(originalTime);
  previewPlaybackSeek(target);
  if (playbackSeekTimer) window.clearTimeout(playbackSeekTimer);
  playbackSeekTimer = window.setTimeout(() => {
    playbackSeekTimer = null;
    commitPlaybackSeek(target);
  }, PLAYBACK_SEEK_DEBOUNCE_MS);
}

function stopPlayback() {
  if (!audioPlayer) return;
  audioPlayer.pause();
  audioPlayer.currentTime = 0;
  stopStemPlayers();
  pendingPlaybackTime = null;
  playbackMarkerTime = null;
  updatePlaybackControls();
  setPlaybackStatus("Stopped");
}

function togglePlay() {
  if (!audioPlayer) {
    if (currentItem?.audio_url) {
      const start = Number(currentItem.preview_offset_sec || 0);
      const end = start + Number(currentItem.preview_duration_sec || 0);
      playAudioPreview(currentItem.audio_url, start, "Blue window", end);
      return;
    }
    setStatus("No short preview audio available. ffmpeg is required so the browser does not load the full track.");
    return;
  }
  if (audioPlayer.paused) {
    audioPlayer
      .play()
      .then(() => setPlaybackStatus(`Playing from ${fmtTime(audioClipStartSec + audioPlayer.currentTime)}`))
      .catch((err) => setStatus(`Audio play failed: ${err.message}`));
  } else {
    audioPlayer.pause();
    setPlaybackStatus(`Paused at ${fmtTime(audioClipStartSec + audioPlayer.currentTime)}`);
  }
}

function waveformExportParams(format) {
  if (!currentItem || !waveDuration) return;
  const wrapperWidth = $("waveWrapper").clientWidth || 1000;
  const dpr = Math.max(2, window.devicePixelRatio || 1);
  const isSvg = format === "svg";
  const params = new URLSearchParams({
    id: currentItem.id,
    start: viewportStart.toFixed(9),
    end: viewportEnd.toFixed(9),
    width: String(Math.max(isSvg ? 9600 : 4800, Math.round(wrapperWidth * dpr * (isSvg ? 8 : 4)))),
    height: isSvg ? "1000" : "1000",
  });
  if (userPick !== null) params.set("user_pick", userPick.toFixed(9));
  const times = markerTimes();
  if (times.micro !== null) params.set("refined_pick", times.micro.toFixed(9));
  if (times.attack !== null) params.set("attack_time", times.attack.toFixed(9));
  if (times.zero !== null) params.set("zero_time", times.zero.toFixed(9));
  if (times.knee !== null) params.set("knee_time", times.knee.toFixed(9));
  if (times.asd !== null) params.set("asd_time", times.asd.toFixed(9));
  return params;
}

function exportPng() {
  const params = waveformExportParams("png");
  if (!params) return;
  window.open(`/media/waveform_png?${params.toString()}`, "_blank");
}

function exportSvg() {
  const params = waveformExportParams("svg");
  if (!params) return;
  window.open(`/media/waveform_svg?${params.toString()}`, "_blank");
}

function renderState(state) {
  appState = state;
  currentItem = state.current;
  userPick = null;
  pickedCandidate = null;
  refinedPick = null;
  refinedInfo = null;
  pendingAutoPlacePick = null;
  pendingAutoPlaceCandidate = null;
  blueAnchorCandidate = null;
  hoverPlaceTime = null;
  setCorrectionMode(false);
  waveView = "window";
  document.body.classList.toggle("visualFirst", visualFirstMode());
  if ($("placeBtn")) $("placeBtn").textContent = visualFirstMode() ? "PLACE 1.1.1" : "PLACE MANUAL";
  if ($("candidateHelp")) {
    $("candidateHelp").textContent = visualFirstMode()
      ? "BLUE ALS is the written impact. ONE and VISUAL are independent checks; ZERO and BODY CREST are diagnostics."
      : "Place or accept the marker, then save.";
  }
  if ($("retrainBtn")) {
    $("retrainBtn").hidden = visualFirstMode();
    $("retrainBtn").disabled = visualFirstMode();
    $("retrainBtn").classList.toggle("disabled", visualFirstMode());
  }
  syncWaveformModeButtons();
  renderAutoAcceptGate(null);
  updateSaveButton();

  if (!currentItem) {
    $("trackName").textContent = "No tracks to review";
    $("trackPath").textContent = "";
    $("queueStats").textContent = "";
    $("drumprintScore").textContent = "--";
    $("structureBeatgrid").textContent = "--";
    $("structureFirstDrop").textContent = "--";
    $("structureSecondDrop").textContent = "--";
    $("fullGrooveScore").textContent = "--";
    $("immediateGrooveScore").textContent = "--";
    $("grooveStability").textContent = "--";
    $("preDropContrast").textContent = "--";
    $("patternStability").textContent = "--";
    $("fakeHitPenalty").textContent = "--";
    $("laterMatchScore").textContent = "--";
    $("microMarker").textContent = "none";
    $("blueMarker").textContent = "none";
    $("microOffset").textContent = "--";
    $("microConfidence").textContent = "--";
    $("attackCleanliness").textContent = "--";
    $("zeroCrossingQuality").textContent = "--";
    renderDropPointContract();
    renderAutoAcceptGate(null);
    destroyWave();
    if ($("candidateList")) $("candidateList").innerHTML = "";
    $("debugImage").style.display = "none";
    $("debugMissing").style.display = "block";
    return;
  }

  $("trackName").textContent = currentItem.track_name;
  $("trackPath").textContent = currentItem.audio_path;
  $("queueStats").textContent =
    `Item ${state.current_index + 1} / ${state.total}\n` +
    `Remaining ${state.counts.remaining} | Approved ${state.counts.approved} | Corrected ${state.counts.corrected} | Skipped ${state.counts.skipped}`;
  updateAiPickDisplay();
  $("confidenceScore").textContent = Number(currentItem.confidence || 0).toFixed(3);
  $("confidenceTier").textContent = currentItem.confidence_tier;
  $("confidenceTier").className = `tier ${tierClass(currentItem.confidence_tier)}`;
  $("selectedBy").textContent = currentItem.selected_by || "--";
  renderStructureSummary();
  renderDropPointContract();
  $("fullGrooveScore").textContent = Number(currentItem.sustained_full_groove_score || 0).toFixed(3);
  $("immediateGrooveScore").textContent = Number(currentItem.immediate_groove_start_score || 0).toFixed(3);
  $("grooveStability").textContent = Number(currentItem.groove_stability || 0).toFixed(3);
  $("preDropContrast").textContent = Number(currentItem.pre_drop_contrast || 0).toFixed(3);
  $("drumprintScore").textContent = Number(currentItem.drumprint_pattern_score || 0).toFixed(3);
  $("patternStability").textContent = Number(currentItem.post_drop_pattern_stability || 0).toFixed(3);
  $("fakeHitPenalty").textContent = Number(currentItem.fake_hit_penalty || 0).toFixed(3);
  $("laterMatchScore").textContent = Number(currentItem.later_drop_match_score || 0).toFixed(3);
  renderAutoAcceptGate(primaryAutoGate(currentItem.auto_accept));

  renderCandidates(currentItem);
  loadWaveform(currentItem);

  if (currentItem.debug_url) {
    $("debugImage").src = currentItem.debug_url;
    $("debugImage").style.display = "block";
    $("debugMissing").style.display = "none";
  } else {
    $("debugImage").style.display = "none";
    $("debugMissing").style.display = "block";
  }
}

async function refresh() {
  const state = await fetchJson("/api/state");
  renderState(state);
}

async function approve() {
  if (!currentItem) return;
  if (reviewUiBusy()) return;
  const blue = blueMarkerInfo();
  const blueTime = optionalMarkerTime(blue.time);
  if (blueTime === null) {
    setStatus("No Boom-proof blue marker is available to accept for this track.");
    return;
  }
  const blueCandidate = blue.candidate || currentItem.selected_candidate || null;
  setReviewActionInFlight(true);
  setStatus(visualFirstMode() ? `Checking Boom proof for blue marker at ${fmtTime(blueTime)}...` : `Accepting blue marker at ${fmtTime(blueTime)}...`);
  try {
    if (visualFirstMode()) {
      const serverValidation = await validateVisualPlacementOnServer(blueTime, { context: "blue", candidate: blueCandidate });
      if (!serverValidation.valid) {
        setStatus(`Blue accept blocked: ${serverValidation.error || "blue marker is not a validated Boom front edge"}`);
        setPlaybackStatus(`Blue accept blocked at ${fmtTime(blueTime)}`);
        return;
      }
      setStatus(`Accepting validated blue marker at ${fmtTime(blueTime)}...`);
    }
    const data = await fetchJson("/api/approve", {
      method: "POST",
      body: JSON.stringify({
        id: currentItem.id,
        marker_time: visualFirstMode() ? blueTime : undefined,
        picked_candidate: visualFirstMode() ? cloneCandidateForCorrection(blueCandidate, blueTime) : undefined,
      }),
    });
    setStatus("Blue marker accepted and logged.");
    renderState(data.state);
  } catch (err) {
    setStatus(`Accept failed: ${err.message}`);
  } finally {
    setReviewActionInFlight(false);
  }
}

async function acceptBlueMarker() {
  if (visualFirstMode()) {
    await approve();
    return;
  }
  await acceptMarker("grid");
}

async function saveCorrection() {
  if (!currentItem) return;
  if (userPick === null) {
    setStatus("Place a marker first.");
    return;
  }
  if (reviewUiBusy()) return;
  saveInFlight = true;
  updateSaveButton();
  try {
    if (visualFirstMode()) {
      const placement = resolveBoomPlacementTime(userPick);
      if (placement.ok && placement.snapped) {
        setUserPick(placement.time);
        setStatus(`Marker snapped to the Boom front edge at ${fmtTime(userPick)} before saving.`);
      }
      setStatus(`Checking Boom proof before saving ${fmtTime(userPick)}...`);
      const serverValidation = await validateVisualPlacementOnServer(userPick);
      if (!serverValidation.valid) {
        const evidence = placement.evidence || {};
        const postPct = Math.round((Number(evidence.normalizedPost) || 0) * 100);
        const peakPct = Math.round((Number(evidence.normalizedPeak) || 0) * 100);
        const localReason = placement.ok ? "" : ` Local view: ${evidence.reason}, body ${postPct}%, peak ${peakPct}%.`;
        setStatus(
          `Save blocked: ${fmtTime(userPick)} is not a validated Boom front edge. ` +
            `${serverValidation.error || "Server Boom proof rejected the marker."}${localReason}`,
        );
        setPlaybackStatus(`Save blocked at ${fmtTime(userPick)}`);
        return;
      }
    }
    const candidateForLearning = visualFirstMode() ? null : pickedCandidate || closestCandidateNearTime(userPick, 0.12);
    const learningCandidate = candidateForLearning ? cloneCandidateForCorrection(candidateForLearning, userPick) : null;
    const learningRank = learningCandidate?.picked_candidate_rank || learningCandidate?.rank || null;
    const saveLabel = learningRank ? `candidate #${learningRank}` : "placed marker";
    setStatus(`Saving ${saveLabel} at ${fmtTime(userPick)}...`);
    const data = await fetchJson("/api/correct", {
      method: "POST",
      body: JSON.stringify({
        id: currentItem.id,
        user_pick: userPick,
        reviewed_from: learningCandidate ? "web_candidate_pick" : "web_manual_marker",
        picked_candidate: learningCandidate,
        top_10_candidates: currentItem.top_10_candidates || [],
      }),
    });
    setStatus(data.regeneration || "Correction logged.");
    renderState(data.state);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`);
  } finally {
    saveInFlight = false;
    updateSaveButton();
  }
}

async function refineMarker() {
  if (!currentItem) return;
  if (reviewUiBusy()) return;
  setStatus("Running sample-level marker refinement...");
  const marker = userPick === null ? activeCenterTime() : userPick;
  const data = await fetchJson("/api/refine_marker", {
    method: "POST",
    body: JSON.stringify({ id: currentItem.id, marker_time: marker }),
  });
  setRefinedPick(data.microalign);
  renderAutoAcceptGate(primaryAutoGate(data.auto_accept));
  setViewportAround(refinedPick, Math.min(markerInspectionRadius(), viewportDuration() / 2), waveView);
  setStatus(
    `Suggested refined marker: ${fmtTime(refinedPick)}\n` +
      `Snap offset: ${Number(data.microalign.snap_offset_ms || 0).toFixed(2)} ms\n` +
      `Micro confidence: ${Number(data.microalign.micro_confidence || 0).toFixed(3)}\n` +
      `Knee marker: ${data.microalign.visual_onset_knee_used ? fmtTime(data.microalign.visual_onset_knee_time) : "--"}\n` +
      `${data.microalign.reason || ""}`,
  );
}

async function autoPlace() {
  if (!currentItem) return;
  if (reviewUiBusy()) return;
  const mode = visualFirstMode() ? "visual_only" : "normal";
  setStatus(
    visualFirstMode()
      ? "Running visual waveform scan, then zoomed marker placement..."
      : "Running full-track AI drop scan, then MicroSnap...",
  );
  const data = await fetchJson("/api/auto_place", {
    method: "POST",
    body: JSON.stringify({ id: currentItem.id, mode }),
  });
  const suggestion = data.suggestion || {};
  if (data.structure_map) {
    currentItem.structure_map = data.structure_map;
    currentItem.beatgrid = data.structure_map.beatgrid || currentItem.beatgrid;
  } else if (data.source_info?.structure_map) {
    currentItem.structure_map = data.source_info.structure_map;
    currentItem.beatgrid = data.source_info.structure_map.beatgrid || currentItem.beatgrid;
  }
  if (Array.isArray(data.candidates)) {
    currentItem.top_10_candidates = data.candidates;
    pickedCandidate = null;
    renderCandidates(currentItem);
  }
  if (suggestion.candidate) {
    currentItem.selected_candidate = suggestion.candidate;
    currentItem.selected_by = suggestion.candidate.selected_by || data.source || currentItem.selected_by;
    $("selectedBy").textContent = currentItem.selected_by || "--";
  }
  const autoPlaceMarker = optionalMarkerTime(suggestion.suggested_time) || candidateTime(suggestion.candidate);
  pendingAutoPlacePick = autoPlaceMarker === null ? null : Number(autoPlaceMarker);
  pendingAutoPlaceCandidate = pendingAutoPlacePick === null ? null : suggestion.candidate || null;
  blueAnchorCandidate = pendingAutoPlaceCandidate;
  if (suggestion.microalign) setRefinedPick(suggestion.microalign);
  else updateSaveButton();
  renderStructureSummary();
  scheduleDrawWaveform();
  renderAutoAcceptGate(primaryAutoGate(suggestion.auto_accept));
  if (refinedPick !== null) setViewportAround(refinedPick, Math.min(markerInspectionRadius(), viewportDuration() / 2), waveView);
  const barPriorText = formatBarPrior(suggestion.bar_prior || barPriorForCandidate(suggestion.candidate));
  const clockText = formatBpmClock(suggestion.bpm_clock || bpmClockForCandidate(suggestion.candidate));
  const first = structureCandidate("first_drop");
  const second = structureCandidate("second_drop");
  const placementLabel =
    suggestion.mode === "visual_only" || data.source === "visual_gui_only"
      ? "visual marker"
      : suggestion.mode === "historical" || data.source === "historical_human_marker"
      ? "historical marker"
      : suggestion.suggested_time !== undefined && suggestion.suggested_time !== null
        ? "suggested for review"
        : "review recommended";
  setStatus(
    `Auto place: ${placementLabel}\n` +
      `Source: ${data.source || "saved_candidates"}\n` +
      `Suggested marker: ${suggestion.suggested_time !== undefined && suggestion.suggested_time !== null ? fmtTime(suggestion.suggested_time) : "--"}\n` +
      `Even-bar prior: ${barPriorText || "--"}\n` +
      `BPM clock: ${clockText || "--"}\n` +
      `First drop: ${first ? formatStructureMarker(first) : "--"}\n` +
      `Second drop: ${second ? formatStructureMarker(second) : "--"}\n` +
      `Reason: ${suggestion.reason || ""}\n` +
      `Knee marker: ${suggestion.microalign?.visual_onset_knee_used ? fmtTime(suggestion.microalign.visual_onset_knee_time) : "--"}\n` +
      `MicroSnap: ${suggestion.microalign?.reason || ""}`,
  );
}

async function acceptMarker(kind) {
  if (!currentItem) return;
  if (reviewUiBusy()) return;
  if (visualFirstMode()) {
    if (kind === "ai" || kind === "grid") {
      await approve();
    } else {
      setStatus("Visual-first review only accepts the Boom-proof blue marker or a saved green placement.");
    }
    return;
  }
  const hasPendingAutoPlace = kind === "ai" && optionalMarkerTime(pendingAutoPlacePick) !== null;
  if (kind === "ai" && !hasPendingAutoPlace) {
    await approve();
    return;
  }
  const acceptedTime = markerTime(kind);
  if (acceptedTime === null) return;
  const blue = kind === "grid" ? blueMarkerInfo() : null;
  const label = kind === "grid" && blue?.label ? `BLUE ${blue.label}` : String(kind || "marker").toUpperCase();
  const reviewedFrom = kind === "grid" ? "web_accept_blue_marker" : hasPendingAutoPlace ? "web_accept_blue_marker" : `web_accept_${kind}_marker`;
  const payload = {
    id: currentItem.id,
    user_pick: acceptedTime,
    reviewed_from: reviewedFrom,
  };
  if (hasPendingAutoPlace) {
    payload.picked_candidate = cloneCandidateForCorrection(pendingAutoPlaceCandidate || currentItem.selected_candidate, acceptedTime);
    payload.top_10_candidates = currentItem.top_10_candidates || [];
  } else if (kind === "grid") {
    payload.picked_candidate = cloneCandidateForCorrection(pickedCandidate || blue?.candidate || defaultBlueAnchorCandidate(), acceptedTime);
    payload.top_10_candidates = currentItem.top_10_candidates || [];
  }
  setReviewActionInFlight(true);
  setStatus(`Accepting ${label} marker at ${fmtTime(acceptedTime)}...`);
  try {
    const data = await fetchJson("/api/correct", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setStatus(data.regeneration || `${label} marker accepted at ${fmtTime(acceptedTime)}.`);
    clearRefinedPick();
    renderState(data.state);
  } catch (err) {
    setStatus(`${label} accept failed: ${err.message}`);
  } finally {
    setReviewActionInFlight(false);
  }
}

async function skip() {
  if (!currentItem) return;
  if (reviewUiBusy()) return;
  setReviewActionInFlight(true);
  setStatus(`Skipping ${currentItem.track_name || "track"}...`);
  try {
    const data = await fetchJson("/api/skip", {
      method: "POST",
      body: JSON.stringify({ id: currentItem.id }),
    });
    setStatus("Skipped.");
    renderState(data.state);
  } catch (err) {
    setStatus(`Skip failed: ${err.message}`);
  } finally {
    setReviewActionInFlight(false);
  }
}

async function navigate(direction) {
  if (reviewUiBusy()) return;
  const data = await fetchJson("/api/navigate", {
    method: "POST",
    body: JSON.stringify({ direction }),
  });
  setStatus("");
  renderState(data);
}

async function retrain() {
  if (visualFirstMode()) {
    setStatus("Retrain is disabled in visual-first review. Use Boom-profile training outside the WebGUI.");
    return;
  }
  setStatus("Training candidate model and running promotion gate...");
  const data = await fetchJson("/api/retrain", { method: "POST", body: "{}" });
  setStatus(data);
  await refresh();
}

function placeMode() {
  if (reviewUiBusy()) return;
  const blockReason = visualDirectPlaceBlockReason();
  if (blockReason === "loading_boom_mask") {
    setStatus("Waiting for the current Boom mask before placing 1.1.1.");
    scheduleWaveformTile(0);
    return;
  }
  if (blockReason === "no_boom_front_edge") {
    setStatus("No hard Boom front edge is visible here. Zoom or move to the first dark drop body edge before placing 1.1.1.");
    return;
  }
  if (visualFirstMode() && currentItem && waveDuration && !visualDirectPlaceReady()) {
    const focus =
      optionalMarkerTime(userPick) ??
      candidateTime(blueAnchorCandidate) ??
      blueMarkerInfo().time ??
      candidateTime(currentItem.selected_candidate) ??
      optionalMarkerTime(currentItem.ai_pick) ??
      activeCenterTime();
    setViewportAround(focus, VISUAL_DIRECT_PLACE_VIEW_SECONDS / 2, "custom");
    setCorrectionMode(false);
    setStatus(`Zoomed to ${fmtTime(focus)}. Click waveform to play/seek. Press PLACE 1.1.1 again when you are ready to mark.`);
    return;
  }
  setCorrectionMode(true);
  setStatus("Click the waveform once to place the true 1.1.1. After that, waveform clicks seek playback again.");
}

function handlePlaybackSeekInput(event) {
  const target = Number(event.target.value);
  if (!Number.isFinite(target)) return;
  schedulePlaybackSeek(target);
}

function handlePlaybackSeekChange(event) {
  const target = Number(event.target.value);
  if (!Number.isFinite(target)) return;
  commitPlaybackSeek(target);
}

function attachEvents() {
  const on = (id, eventName, handler, options) => {
    const target = $(id);
    if (target) target.addEventListener(eventName, handler, options);
  };
  $("detailPanel")?.addEventListener("toggle", () => {
    if (!syncingDetailPanel) detailPanelUserChanged = true;
  });
  on("approveBtn", "click", () => acceptMarker("ai"));
  on("acceptKneeBtn", "click", () => acceptMarker("knee"));
  on("acceptGridBtn", "click", acceptBlueMarker);
  on("placeBtn", "click", placeMode);
  on("aiRefineBtn", "click", refineMarker);
  on("aiAutoPlaceBtn", "click", autoPlace);
  on("saveCorrectionBtn", "click", saveCorrection);
  on("clearMarkerBtn", "click", clearUserPick);
  on("skipBtn", "click", skip);
  on("retrainBtn", "click", retrain);
  on("wavePlayBtn", "pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  on("wavePlayBtn", "click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    togglePlay();
  });
  on("boomWaveBtn", "click", () => setWaveformVisualMode("boom"));
  on("rmsWaveBtn", "click", () => setWaveformVisualMode("rms"));
  on("peakWaveBtn", "click", () => setWaveformVisualMode("peaks"));
  $("playViewBtn").addEventListener("click", playCurrentZoom);
  $("playMarkerBtn").addEventListener("click", playAroundMarker);
  $("stopBtn").addEventListener("click", stopPlayback);
  $("fullTrackBtn").addEventListener("click", () => applyPreset("full"));
  $("aiWindowBtn").addEventListener("click", () => applyPreset("window"));
  $("jumpAiBtn").addEventListener("click", () => {
    const blue = blueMarkerInfo().time;
    if (blue === null) {
      setStatus("No Boom-proof blue marker is available for this track.");
      return;
    }
    scrollToOriginalTime(blue);
  });
  $("exportPngBtn").addEventListener("click", exportPng);
  $("exportSvgBtn").addEventListener("click", exportSvg);
  $("zoomOutBtn").addEventListener("click", () => zoomBy(-1));
  $("zoomInBtn").addEventListener("click", () => zoomBy(1));
  $("zoomResetBtn").addEventListener("click", resetZoom);
  $("zoomSlider").addEventListener("input", (event) => setZoomFromSlider(Number(event.target.value)));
  $("playSeek").addEventListener("input", handlePlaybackSeekInput);
  $("playSeek").addEventListener("change", handlePlaybackSeekChange);

  document.querySelectorAll(".presetBtn").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.getAttribute("data-preset")));
  });

  $("waveWrapper").addEventListener(
    "click",
    (event) => {
      if (!correctionMode || !currentItem || !waveDuration) return;
      event.preventDefault();
      event.stopPropagation();
      placeAtClientX(event.clientX);
    },
    { capture: true },
  );

  $("waveWrapper").addEventListener(
    "dblclick",
    (event) => {
      if (!visualFirstMode() || correctionMode || !currentItem || !waveDuration) return;
      event.preventDefault();
      event.stopPropagation();
      suppressWaveClick = true;
      visualZoomAtClientX(event.clientX);
    },
    { capture: true },
  );

  $("waveWrapper").addEventListener("click", (event) => {
    if (correctionMode || !currentItem || !waveDuration) return;
    if (suppressWaveClick) {
      suppressWaveClick = false;
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    event.preventDefault();
    if (visualFirstMode()) {
      playPlaybackFromClientX(event.clientX);
      return;
    }
    if (isMobileLayout()) playPlaybackFromClientX(event.clientX);
    else seekPlaybackToClientX(event.clientX);
  });

  $("waveWrapper").addEventListener("mousemove", (event) => {
    if (!visualDirectPlaceReady() || !correctionMode || userPick !== null) {
      if (hoverPlaceTime !== null) {
        hoverPlaceTime = null;
        scheduleDrawWaveform();
      }
      return;
    }
    const next = timeFromClientX(event.clientX);
    const placement = resolveBoomPlacementTime(next);
    if (!placement.ok) {
      if (hoverPlaceTime !== null) {
        hoverPlaceTime = null;
        scheduleDrawWaveform();
      }
      return;
    }
    if (hoverPlaceTime === null || Math.abs(Number(placement.time) - Number(hoverPlaceTime)) > minViewSeconds()) {
      hoverPlaceTime = placement.time;
      scheduleDrawWaveform();
    }
  });

  $("waveWrapper").addEventListener("mouseleave", () => {
    if (hoverPlaceTime !== null) {
      hoverPlaceTime = null;
      scheduleDrawWaveform();
    }
  });

  $("waveWrapper").addEventListener(
    "wheel",
    (event) => {
      if (!currentItem || !waveDuration) return;
      const absX = Math.abs(Number(event.deltaX || 0));
      const absY = Math.abs(Number(event.deltaY || 0));
      const isPinchZoom = Boolean(event.ctrlKey || event.metaKey);
      const isHorizontalPan = absX > absY || event.shiftKey;
      if (isPinchZoom || !isHorizontalPan) {
        event.preventDefault();
        const dy = normalizedWheelDelta(event.deltaY, event);
        setZoomFactor(Math.exp(dy * WHEEL_ZOOM_SENSITIVITY), event.clientX);
      } else if (isHorizontalPan) {
        event.preventDefault();
        const raw = event.shiftKey && absX <= absY ? event.deltaY : event.deltaX;
        panBy(normalizedWheelDelta(raw, event));
      }
    },
    { passive: false },
  );

  $("waveWrapper").addEventListener("gesturestart", (event) => {
    if (!currentItem || !waveDuration) return;
    if (activePointers.size >= 2) return;
    event.preventDefault();
    const anchorClientX = markerAnchorClientX(event.clientX);
    gestureState = {
      anchorTime: timeFromClientX(anchorClientX),
      startSpan: viewportDuration(),
      anchorClientX,
    };
  });

  $("waveWrapper").addEventListener("gesturechange", (event) => {
    if (!gestureState || !currentItem || !waveDuration) return;
    if (activePointers.size >= 2) return;
    event.preventDefault();
    const scale = Math.max(0.05, Number(event.scale || 1));
    const factor = acceleratedZoomFactor(1 / scale);
    const anchorClientX = markerAnchorClientX(event.clientX || gestureState.anchorClientX);
    setViewportWithAnchor(gestureState.startSpan * factor, gestureState.anchorTime, anchorClientX);
  });

  $("waveWrapper").addEventListener("gestureend", () => {
    gestureState = null;
  });

  $("waveWrapper").addEventListener("pointerdown", (event) => {
    if (event.pointerType === "mouse") return;
    activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (!correctionMode && activePointers.size === 1) {
      waveDragState = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        lastX: event.clientX,
        moved: false,
      };
      $("waveWrapper").setPointerCapture?.(event.pointerId);
    }
    if (activePointers.size === 2) {
      waveDragState = null;
      const points = [...activePointers.values()];
      const centerX = pointerMidpointClientX(points);
      pinchState = {
        startDistance: Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y),
        startSpan: viewportDuration(),
        anchorTime: timeFromClientX(centerX),
      };
      $("waveWrapper").setPointerCapture?.(event.pointerId);
    }
  });

  $("waveWrapper").addEventListener("pointermove", (event) => {
    if (!activePointers.has(event.pointerId)) return;
    activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (!pinchState && waveDragState && waveDragState.pointerId === event.pointerId) {
      const totalX = Number(event.clientX) - waveDragState.startX;
      const totalY = Number(event.clientY) - waveDragState.startY;
      const horizontalDrag = Math.abs(totalX) > 8 && Math.abs(totalX) > Math.abs(totalY) * 1.15;
      if (horizontalDrag || waveDragState.moved) {
        event.preventDefault();
        const dx = Number(event.clientX) - waveDragState.lastX;
        waveDragState.lastX = Number(event.clientX);
        waveDragState.moved = true;
        panBy(-dx);
      }
      return;
    }
    if (!pinchState || activePointers.size < 2) return;
    event.preventDefault();
    const points = [...activePointers.values()];
    const distance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
    const centerX = pointerMidpointClientX(points);
    if (pinchState.startDistance > 0 && distance > 0) {
      const factor = acceleratedZoomFactor(pinchState.startDistance / distance);
      setViewportWithAnchor(pinchState.startSpan * factor, pinchState.anchorTime, centerX);
    }
  });

  const clearPointer = (event) => {
    if (waveDragState && waveDragState.pointerId === event.pointerId) {
      suppressWaveClick = Boolean(waveDragState.moved);
      waveDragState = null;
    }
    activePointers.delete(event.pointerId);
    if (activePointers.size < 2) pinchState = null;
  };
  $("waveWrapper").addEventListener("pointerup", clearPointer);
  $("waveWrapper").addEventListener("pointercancel", clearPointer);
  $("waveWrapper").addEventListener("pointerleave", clearPointer);

  window.addEventListener("resize", () => {
    syncDetailPanelForViewport();
    scheduleDrawWaveform();
    scheduleWaveformTile(RESIZE_TILE_DEBOUNCE_MS);
  });

  window.addEventListener("keydown", (event) => {
    if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) return;
    const key = event.key.toLowerCase();
    if (key === "y" || key === "b") acceptBlueMarker();
    if (key === "n") placeMode();
    if (key === "s") skip();
    if (key === "r" && !visualFirstMode()) retrain();
    if (key === "c") clearUserPick();
    if (key === "=" || key === "+") zoomBy(1);
    if (key === "-" || key === "_") zoomBy(-1);
    if (key === "0") resetZoom();
    if (event.code === "Space") {
      event.preventDefault();
      togglePlay();
    }
  });
}

attachEvents();
syncDetailPanelForViewport();
refresh().catch((err) => setStatus(`Failed to load state: ${err.message}`));
