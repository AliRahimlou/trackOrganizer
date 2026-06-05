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
let refinedPick = null;
let refinedInfo = null;
let pendingAutoPlacePick = null;
let pendingAutoPlaceCandidate = null;
let waveDuration = 0;
let viewportStart = 0;
let viewportEnd = 1;
let waveTile = null;
let waveTiles = [];
let waveView = "window";
let waveformVisualMode = window.localStorage?.getItem("dropReviewWaveformVisualMode") === "peaks" ? "peaks" : "rms";
let tileRequestSeq = 0;
let tileRequestTimer = null;
let tileAbortController = null;
let drawFrameId = null;
let pinchState = null;
let gestureState = null;
let waveDragState = null;
let suppressWaveClick = false;
let detailPanelUserChanged = false;
let syncingDetailPanel = false;
const activePointers = new Map();
const DESKTOP_WAVE_LANE_HEIGHT = 112;
const MOBILE_WAVE_LANE_HEIGHT = 122;
const MIN_DESKTOP_WAVE_HEIGHT = 240;
const MIN_MOBILE_WAVE_HEIGHT = 260;
const DEFAULT_WINDOW_BEFORE = 20;
const DEFAULT_WINDOW_AFTER = 30;
const MIN_VIEW_SAMPLES = 2;
const WAVEFORM_DETAIL_MULTIPLIER = 8;
const WAVEFORM_MAX_TILE_BINS = 48000;
const HYPER_RMS_TARGET = 0.78;
const HYPER_RMS_CEILING = 0.985;
const HYPER_RMS_KNEE = 0.72;
const HYPER_RMS_MIN_MAKEUP = 0.85;
const HYPER_RMS_MAX_MAKEUP = 7.5;
const HYPER_RMS_POWER = 0.82;
const HYPER_RMS_PEAK_GHOST_ALPHA = 0.06;
const HYPER_RMS_MIN_SMOOTH_SECONDS = 0.006;
const HYPER_RMS_MAX_SMOOTH_SECONDS = 0.120;
const RMS_INSPECTION_MIN_SAMPLE_SPACING = 0.10;
const RMS_INSPECTION_ZERO_CROSSING_SPACING = 0.65;
const RMS_INSPECTION_PEAK_ALPHA = 0.30;
const RMS_DROP_INSPECTION_RADIUS_SECONDS = 0.006;
const TILE_REQUEST_DEBOUNCE_MS = 60;
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

function availableStemCount() {
  const loaded = usableWaveTiles().length;
  if (loaded) return loaded;
  const itemStems = (currentItem?.stem_waveforms || []).filter((stem) => stem.available !== false).length;
  return Math.max(1, itemStems || 1);
}

function waveCanvasHeight() {
  const laneHeight = isMobileLayout() ? MOBILE_WAVE_LANE_HEIGHT : DESKTOP_WAVE_LANE_HEIGHT;
  const minHeight = isMobileLayout() ? MIN_MOBILE_WAVE_HEIGHT : MIN_DESKTOP_WAVE_HEIGHT;
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

function barPriorForCandidate(candidate) {
  return candidate?.musical_bar_prior || candidate?.bar_prior || null;
}

function bpmClockForCandidate(candidate) {
  return candidate?.bpm_clock || candidate?.clock || null;
}

function structureMap() {
  return currentItem?.structure_map || null;
}

function structureBeatgrid() {
  return structureMap()?.beatgrid || currentItem?.beatgrid || null;
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

function markerTimes() {
  return {
    ai: optionalMarkerTime(pendingAutoPlacePick) || optionalMarkerTime(currentItem?.ai_pick),
    knee: kneeMarkerTime(),
    manual: userPick === null ? null : optionalMarkerTime(userPick),
  };
}

function markerTime(kind) {
  return markerTimes()[kind] || null;
}

function updateMarkerAcceptButton(id, label, time) {
  const button = $(id);
  if (!button) return;
  const enabled = time !== null;
  button.classList.toggle("disabled", !enabled);
  button.textContent = enabled ? `ACCEPT ${label} ${fmtTime(time)}` : `NO ${label} MARKER`;
}

function updateAiPickDisplay() {
  const target = $("aiPick");
  if (!target) return;
  const time = markerTimes().ai;
  target.textContent = time === null ? "--" : `${fmtTime(time)} (${time.toFixed(6)}s)`;
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
  saveButton.classList.toggle("disabled", !hasPick || saveInFlight);
  saveButton.textContent = saveInFlight
    ? "SAVING..."
    : times.manual === null
      ? "SAVE PLACED"
      : `SAVE ${fmtTime(times.manual)}`;
  $("clearMarkerBtn").classList.toggle("disabled", !hasPick || saveInFlight);
  updateMarkerAcceptButton("approveBtn", "AI", times.ai);
  updateMarkerAcceptButton("acceptKneeBtn", "KNEE", times.knee);
  updateAiPickDisplay();
  $("userPick").textContent = times.manual === null ? "none" : `${fmtTime(times.manual)} (${times.manual.toFixed(6)}s)`;
  $("microMarker").textContent = times.knee === null ? "none" : `${fmtTime(times.knee)} (${times.knee.toFixed(6)}s)`;
  updateMetric("microOffset", micro?.snap_offset_ms, (value) => `${value.toFixed(2)} ms`);
  updateMetric("microConfidence", micro?.micro_confidence, (value) => value.toFixed(3));
  updateMetric("attackCleanliness", micro?.attack_cleanliness, (value) => value.toFixed(3));
  updateMetric("zeroCrossingQuality", micro?.zero_crossing_quality, (value) => value.toFixed(3));
  drawWaveform();
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

function clearUserPick() {
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
  const beatPx = beatSeconds * pixelsPerSecond();
  const firstBeat = Math.max(0, Math.floor(viewportStart / beatSeconds) - 1);
  const lastBeat = Math.ceil(viewportEnd / beatSeconds) + 1;
  const maxBeats = 1400;
  if (lastBeat - firstBeat > maxBeats) return;

  ctx.save();
  ctx.font = "10px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  for (let beat = firstBeat; beat <= lastBeat; beat += 1) {
    const time = beat * beatSeconds;
    const x = timeToX(time);
    if (x < -30 || x > width + 30) continue;
    const isOne = beat % 4 === 0;
    const isPhrase = beat % 64 === 0;
    const isEightBar = beat % 32 === 0;
    ctx.setLineDash([]);
    ctx.strokeStyle = isPhrase
      ? "rgba(15,111,191,0.62)"
      : isEightBar
        ? "rgba(15,111,191,0.45)"
        : isOne
          ? "rgba(15,111,191,0.32)"
          : "rgba(15,111,191,0.12)";
    ctx.lineWidth = isPhrase ? 2.2 : isEightBar ? 1.8 : isOne ? 1.35 : 0.75;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, 0);
    ctx.lineTo(Math.round(x) + 0.5, height);
    ctx.stroke();

    if (isOne && (beatPx >= 42 || isPhrase || isEightBar)) {
      const barNumber = Math.floor(beat / 4) + 1;
      const label = `${Math.round(bpm)} BPM ONE b${barNumber}`;
      const labelWidth = Math.ceil(ctx.measureText(label).width) + 8;
      ctx.fillStyle = "rgba(255,255,255,0.88)";
      ctx.fillRect(x + 4, 24, labelWidth, 15);
      ctx.fillStyle = "#0a4e88";
      ctx.fillText(label, x + 8, 35);
    } else if (!isOne && beatPx >= 90) {
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

function waveformGain(tile = waveTile) {
  return 0.92 / Math.max(Number(tile?.amplitude_peak || 1), 1e-9);
}

function sampleY(sample, laneTop, laneHeight, tile = waveTile) {
  return laneTop + laneHeight / 2 - Math.max(-1, Math.min(1, Number(sample) * waveformGain(tile))) * (laneHeight * 0.40);
}

function rmsVisualCeiling(tile = waveTile) {
  const ceiling = Number(tile?.rms_visual_ceiling || tile?.rms_percentile_95 || tile?.rms_percentile_99 || tile?.rms_peak || 0);
  if (Number.isFinite(ceiling) && ceiling > 1e-6) return ceiling;
  const rms = Array.isArray(tile?.rms) ? tile.rms.map(Number).filter((value) => Number.isFinite(value) && value > 0) : [];
  if (!rms.length) return Math.max(Number(tile?.amplitude_peak || 1), 1e-6);
  rms.sort((a, b) => a - b);
  return Math.max(rms[Math.floor(rms.length * 0.95)] || rms[rms.length - 1] || 1, 1e-6);
}

function rmsMaximizerMakeup(tile = waveTile) {
  const explicit = Number(tile?.visual_maximizer_makeup_gain);
  if (Number.isFinite(explicit) && explicit > 0) {
    return Math.max(HYPER_RMS_MIN_MAKEUP, Math.min(HYPER_RMS_MAX_MAKEUP, explicit));
  }
  const p95 = Number(tile?.rms_percentile_95 || rmsVisualCeiling(tile));
  const reference = Math.max(p95, 1e-6);
  return Math.max(HYPER_RMS_MIN_MAKEUP, Math.min(HYPER_RMS_MAX_MAKEUP, HYPER_RMS_TARGET / reference));
}

function softLimitMaximizedRms(value, tile = waveTile) {
  const makeup = rmsMaximizerMakeup(tile);
  const ceiling = Math.max(0.5, Math.min(1, Number(tile?.visual_maximizer_ceiling || HYPER_RMS_CEILING)));
  const knee = Math.max(0.2, Math.min(ceiling - 0.02, Number(tile?.visual_maximizer_knee || HYPER_RMS_KNEE)));
  const lifted = Math.max(0, Number(value) || 0) * makeup;
  if (lifted <= knee) return lifted;
  const range = Math.max(0.001, ceiling - knee);
  return knee + range * Math.tanh((lifted - knee) / range);
}

function hyperRmsAmplitude(value, tile = waveTile) {
  const limited = softLimitMaximizedRms(value, tile);
  const ceiling = Math.max(0.5, Math.min(1, Number(tile?.visual_maximizer_ceiling || HYPER_RMS_CEILING)));
  return Math.max(0, Math.min(1, Math.pow(limited / ceiling, HYPER_RMS_POWER)));
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

function rmsY(value, laneTop, laneHeight, tile = waveTile, polarity = 1) {
  const amp = hyperRmsAmplitude(value, tile);
  return laneTop + laneHeight / 2 - polarity * amp * (laneHeight * 0.42);
}

function waveformModeText() {
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
  return waveformVisualMode === "rms" ? RMS_DROP_INSPECTION_RADIUS_SECONDS : 0.05;
}

function setWaveformVisualMode(mode) {
  waveformVisualMode = mode === "peaks" ? "peaks" : "rms";
  try {
    window.localStorage?.setItem("dropReviewWaveformVisualMode", waveformVisualMode);
  } catch (_) {
    // Local storage can be unavailable in restricted browser contexts.
  }
  syncWaveformModeButtons();
  updateAudioStatus();
  scheduleDrawWaveform();
}

function syncWaveformModeButtons() {
  $("rmsWaveBtn")?.classList.toggle("active", waveformVisualMode === "rms");
  $("peakWaveBtn")?.classList.toggle("active", waveformVisualMode === "peaks");
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
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = span / Math.max(1, mins.length);
  const strokeWidth = Math.max(0.55, Math.min(4, binSpan * pixelsPerSecond()));
  const detailRatio = mins.length / Math.max(width, 1);

  ctx.beginPath();
  for (let i = 0; i < maxs.length; i += 1) {
    const t = start + (i + 0.5) * binSpan;
    const x = timeToX(t);
    const y = sampleY(maxs[i], laneTop, laneHeight, tile);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  for (let i = mins.length - 1; i >= 0; i -= 1) {
    const t = start + (i + 0.5) * binSpan;
    ctx.lineTo(timeToX(t), sampleY(mins[i], laneTop, laneHeight, tile));
  }
  ctx.closePath();
  ctx.fillStyle = stemFill(tile);
  ctx.fill();

  ctx.save();
  ctx.globalAlpha = detailRatio > 2 ? 0.62 : 0.92;
  ctx.strokeStyle = stemStroke(tile);
  ctx.lineWidth = strokeWidth;
  for (let i = 0; i < mins.length; i += 1) {
    const t = start + (i + 0.5) * binSpan;
    const x = timeToX(t);
    if (x < -strokeWidth || x > width + strokeWidth) continue;
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
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = span / Math.max(1, mins.length);
  ctx.save();
  ctx.globalAlpha = RMS_INSPECTION_PEAK_ALPHA;
  ctx.strokeStyle = stemStroke(tile);
  ctx.lineWidth = Math.max(0.75, Math.min(1.8, binSpan * pixelsPerSecond()));
  ctx.beginPath();
  for (let i = 0; i < maxs.length; i += 1) {
    const t = start + (i + 0.5) * binSpan;
    const x = timeToX(t);
    const y = sampleY(maxs[i], laneTop, laneHeight, tile);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i < mins.length; i += 1) {
    const t = start + (i + 0.5) * binSpan;
    const x = timeToX(t);
    const y = sampleY(mins[i], laneTop, laneHeight, tile);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawSampleInspectionOverlay(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const samples = tile?.samples || [];
  if (samples.length < 2) return false;
  const sr = Number(tile?.sample_rate || sampleRate());
  const startSample = Number(tile.sample_start || tile.start_sample || 0);
  const visibleStart = Math.max(0, Math.floor(viewportStart * sr) - startSample - 2);
  const visibleEnd = Math.min(samples.length, Math.ceil(viewportEnd * sr) - startSample + 2);
  if (visibleEnd - visibleStart < 2) return false;

  const spacing = pixelsPerSecond() / Math.max(sr, 1);
  if (spacing < RMS_INSPECTION_MIN_SAMPLE_SPACING) return false;

  const points = [];
  for (let index = visibleStart; index < visibleEnd; index += 1) {
    const time = (startSample + index) / sr;
    const sample = Number(samples[index]);
    points.push([timeToX(time), sampleY(sample, laneTop, laneHeight, tile), sample]);
  }

  ctx.save();
  ctx.strokeStyle = "rgba(17,24,32,0.82)";
  ctx.lineWidth = spacing >= 2 ? 1.35 : 0.85;
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  if (spacing >= RMS_INSPECTION_ZERO_CROSSING_SPACING) {
    const mid = laneTop + laneHeight / 2;
    ctx.strokeStyle = "rgba(17,24,32,0.72)";
    ctx.fillStyle = "rgba(17,24,32,0.72)";
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
  }

  ctx.restore();
  return true;
}

function drawRmsEnvelope(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const rawRms = Array.isArray(tile?.rms) ? tile.rms : [];
  if (rawRms.length < 2) return false;
  const start = tileStartSec(tile);
  const span = Math.max(tileEndSec(tile) - start, 1 / Math.max(sampleRate(), 1));
  const binSpan = span / Math.max(1, rawRms.length);
  const rms = smoothRmsValues(rawRms, binSpan);

  ctx.beginPath();
  for (let i = 0; i < rms.length; i += 1) {
    const t = start + (i + 0.5) * binSpan;
    const x = timeToX(t);
    const y = rmsY(rms[i], laneTop, laneHeight, tile, 1);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  for (let i = rms.length - 1; i >= 0; i -= 1) {
    const t = start + (i + 0.5) * binSpan;
    ctx.lineTo(timeToX(t), rmsY(rms[i], laneTop, laneHeight, tile, -1));
  }
  ctx.closePath();
  ctx.fillStyle = stemFill(tile);
  ctx.fill();

  ctx.save();
  ctx.strokeStyle = stemStroke(tile);
  ctx.lineWidth = 1.45;
  ctx.globalAlpha = 0.88;
  ctx.beginPath();
  for (let i = 0; i < rms.length; i += 1) {
    const t = start + (i + 0.5) * binSpan;
    const x = timeToX(t);
    const y = rmsY(rms[i], laneTop, laneHeight, tile, 1);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i < rms.length; i += 1) {
    const t = start + (i + 0.5) * binSpan;
    const x = timeToX(t);
    const y = rmsY(rms[i], laneTop, laneHeight, tile, -1);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();

  drawPeakBoundaryOverlay(ctx, width, laneTop, laneHeight, tile);
  drawPeakGhost(ctx, width, laneTop, laneHeight, tile);
  drawSampleInspectionOverlay(ctx, width, laneTop, laneHeight, tile);
  return true;
}

function drawSamples(ctx, width, laneTop, laneHeight, tile = waveTile) {
  const samples = tile?.samples || [];
  if (samples.length < 2) return;
  const sr = Number(tile?.sample_rate || sampleRate());
  const startSample = Number(tile.sample_start || tile.start_sample || 0);
  const visibleStart = Math.max(0, Math.floor(viewportStart * sr) - startSample - 2);
  const visibleEnd = Math.min(samples.length, Math.ceil(viewportEnd * sr) - startSample + 2);
  if (visibleEnd - visibleStart < 2) return;
  const points = [];
  for (let index = visibleStart; index < visibleEnd; index += 1) {
    const time = (startSample + index) / sr;
    points.push([timeToX(time), sampleY(samples[index], laneTop, laneHeight, tile), Number(samples[index])]);
  }

  ctx.strokeStyle = stemStroke(tile);
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const spacing = Math.abs(points[1][0] - points[0][0]);
  if (spacing >= 5) {
    ctx.fillStyle = stemStroke(tile);
    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (spacing >= 3) {
    ctx.strokeStyle = "#111820";
    ctx.fillStyle = "#111820";
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
  drawMarker(overlay, currentItem.ai_pick, "ai", "AI");
  drawMarker(overlay, kneeMarkerTime(), "knee", "KNEE");
  (currentItem.top_10_candidates || []).forEach((cand, index) => {
    const labelTop = isMobileLayout() ? 4 + Math.min(index, 9) * 19 : 4;
    drawMarker(overlay, candidateTime(cand), "candidate", `#${cand.rank}`, { labelTop });
  });
  if (userPick !== null) drawMarker(overlay, userPick, "user", "YOU");
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
    li.className = cand.selected || isPicked ? "selected" : "";
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
}

function updateAudioStatus() {
  if (!currentItem || !waveDuration) return;
  const primary = waveTile || usableWaveTiles()[0] || null;
  const stemCount = Math.max(1, usableWaveTiles().length || availableStemCount());
  const mode = waveformVisualMode === "rms" && primary?.rms?.length
    ? waveformModeText()
    : primary?.mode === "samples" ? "raw samples" : "ultra min/max peaks";
  const cache = primary?.cache_hit ? "cache hit" : "cache build";
  const detailCount =
    waveformVisualMode === "rms" && primary?.rms?.length
      ? `${(primary?.rms || []).length} rms bins`
      : primary?.mode === "samples"
      ? `${(primary?.samples || []).length} samples`
      : `${(primary?.mins || []).length} bins`;
  const secPerBin =
    waveformVisualMode === "rms" && primary?.rms?.length
      ? Math.max(0, Number(primary?.end_sec || 0) - Number(primary?.start_sec || 0)) / Math.max((primary?.rms || []).length, 1)
      : primary?.mode === "samples"
      ? 1 / Math.max(sampleRate(), 1)
      : Math.max(0, Number(primary?.end_sec || 0) - Number(primary?.start_sec || 0)) / Math.max((primary?.mins || []).length, 1);
  const maximizerText = waveformVisualMode === "rms" && primary?.rms?.length
    ? ` | max ${rmsMaximizerMakeup(primary).toFixed(2)}x`
    : "";
  const inspectionText = waveformVisualMode === "rms" && primary?.rms?.length
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
  scheduleDrawWaveform();
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
  const ai = Number(currentItem.ai_pick || 0);
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

function placeAtClientX(clientX) {
  if (!correctionMode || !currentItem || !waveDuration) return;
  setUserPick(timeFromClientX(clientX));
  setCorrectionMode(false);
  setStatus(`Correction marker placed at ${fmtTime(userPick)} (${userPick.toFixed(6)}s). Waveform clicks now seek playback.`);
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
  waveView = "window";
  syncViewButtons();
  setupCanvas();
  focusAiWindow();
}

function activeCenterTime() {
  if (userPick !== null) return userPick;
  if (selectedMicroTime() !== null) return selectedMicroTime();
  return Number(currentItem?.ai_pick || 0);
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
  audioPlayer.preload = showReady ? "auto" : "metadata";
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
  startStemPlayers(false);
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
      playAudioPreview(currentItem.audio_url, start, "AI window", end);
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

function exportPng() {
  if (!currentItem || !waveDuration) return;
  const params = new URLSearchParams({
    id: currentItem.id,
    start: viewportStart.toFixed(9),
    end: viewportEnd.toFixed(9),
    width: String(Math.max(2400, Math.round(($("waveWrapper").clientWidth || 1000) * Math.max(2, window.devicePixelRatio || 1) * 2))),
    height: "900",
  });
  if (userPick !== null) params.set("user_pick", userPick.toFixed(9));
  const times = markerTimes();
  if (times.micro !== null) params.set("refined_pick", times.micro.toFixed(9));
  if (times.attack !== null) params.set("attack_time", times.attack.toFixed(9));
  if (times.zero !== null) params.set("zero_time", times.zero.toFixed(9));
  if (times.knee !== null) params.set("knee_time", times.knee.toFixed(9));
  if (times.asd !== null) params.set("asd_time", times.asd.toFixed(9));
  window.open(`/media/waveform_png?${params.toString()}`, "_blank");
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
  setCorrectionMode(false);
  waveView = "window";
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
    $("microOffset").textContent = "--";
    $("microConfidence").textContent = "--";
    $("attackCleanliness").textContent = "--";
    $("zeroCrossingQuality").textContent = "--";
    renderAutoAcceptGate(null);
    destroyWave();
    $("candidateList").innerHTML = "";
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
  const data = await fetchJson("/api/approve", {
    method: "POST",
    body: JSON.stringify({ id: currentItem.id }),
  });
  setStatus("AI marker accepted and logged.");
  renderState(data.state);
}

async function saveCorrection() {
  if (!currentItem) return;
  if (userPick === null) {
    setStatus("Pick a candidate number or place a marker first.");
    return;
  }
  if (saveInFlight) return;
  saveInFlight = true;
  updateSaveButton();
  const saveLabel = pickedCandidate?.picked_candidate_rank ? `candidate #${pickedCandidate.picked_candidate_rank}` : "placed marker";
  setStatus(`Saving ${saveLabel} at ${fmtTime(userPick)}...`);
  try {
    const data = await fetchJson("/api/correct", {
      method: "POST",
      body: JSON.stringify({
        id: currentItem.id,
        user_pick: userPick,
        reviewed_from: pickedCandidate ? "web_candidate_pick" : "web_manual_marker",
        picked_candidate: pickedCandidate,
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
  setStatus("Running full-track AI drop scan, then MicroSnap...");
  const data = await fetchJson("/api/auto_place", {
    method: "POST",
    body: JSON.stringify({ id: currentItem.id, mode: "normal" }),
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
    suggestion.mode === "historical" || data.source === "historical_human_marker"
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
  const hasPendingAutoPlace = kind === "ai" && optionalMarkerTime(pendingAutoPlacePick) !== null;
  if (kind === "ai" && !hasPendingAutoPlace) {
    await approve();
    return;
  }
  const acceptedTime = markerTime(kind);
  if (acceptedTime === null) return;
  const label = String(kind || "marker").toUpperCase();
  const payload = {
    id: currentItem.id,
    user_pick: acceptedTime,
    reviewed_from: hasPendingAutoPlace ? "web_accept_auto_place_marker" : `web_accept_${kind}_marker`,
  };
  if (hasPendingAutoPlace) {
    payload.picked_candidate = cloneCandidateForCorrection(pendingAutoPlaceCandidate || currentItem.selected_candidate, acceptedTime);
    payload.top_10_candidates = currentItem.top_10_candidates || [];
  }
  const data = await fetchJson("/api/correct", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  setStatus(data.regeneration || `${label} marker accepted at ${fmtTime(acceptedTime)}.`);
  clearRefinedPick();
  renderState(data.state);
}

async function skip() {
  if (!currentItem) return;
  const data = await fetchJson("/api/skip", {
    method: "POST",
    body: JSON.stringify({ id: currentItem.id }),
  });
  setStatus("Skipped.");
  renderState(data.state);
}

async function navigate(direction) {
  const data = await fetchJson("/api/navigate", {
    method: "POST",
    body: JSON.stringify({ direction }),
  });
  setStatus("");
  renderState(data);
}

async function retrain() {
  setStatus("Training candidate model and running promotion gate...");
  const data = await fetchJson("/api/retrain", { method: "POST", body: "{}" });
  setStatus(data);
  await refresh();
}

function placeMode() {
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
  on("rmsWaveBtn", "click", () => setWaveformVisualMode("rms"));
  on("peakWaveBtn", "click", () => setWaveformVisualMode("peaks"));
  $("playViewBtn").addEventListener("click", playCurrentZoom);
  $("playMarkerBtn").addEventListener("click", playAroundMarker);
  $("stopBtn").addEventListener("click", stopPlayback);
  $("fullTrackBtn").addEventListener("click", () => applyPreset("full"));
  $("aiWindowBtn").addEventListener("click", () => applyPreset("window"));
  $("jumpAiBtn").addEventListener("click", () => scrollToOriginalTime(currentItem?.ai_pick || 0));
  $("exportPngBtn").addEventListener("click", exportPng);
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

  $("waveWrapper").addEventListener("click", (event) => {
    if (correctionMode || !currentItem || !waveDuration) return;
    if (suppressWaveClick) {
      suppressWaveClick = false;
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    event.preventDefault();
    if (isMobileLayout()) playPlaybackFromClientX(event.clientX);
    else seekPlaybackToClientX(event.clientX);
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
    if (key === "y") acceptMarker("ai");
    if (key === "n") placeMode();
    if (key === "s") skip();
    if (key === "r") retrain();
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
