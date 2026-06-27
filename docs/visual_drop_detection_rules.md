# Visual Drop Detection Rules

The visual-first detector must behave like the review GUI workflow. The target
is a DJ-usable Ableton `1.1.1` marker: the true drop impact, on the musical one,
inside the drums stem.

## Hard Acceptance Gate

Accept a marker only when both requirements are true:

1. It is visually at the start of the hard-hitting, sustained drum/bass drop
   body.
2. It is on the one of the BPM-derived metronome grid.

If the marker is not on a real drop, skip or correct it. If it is on beat two,
beat three, beat four, an off-grid transient, a tail, or a pre-hit, skip or
correct it. Do not approve a marker just because it is close to a candidate,
close to a loud transient, or produced by MicroSnap.

A marker on blank or near-empty waveform is an automatic failure. The BPM grid
can propose a bar start, but it cannot override the visual fact that no drop
body exists at that point. When the blue marker lands on empty space, scan the
same drop region for the first credible drum/bass body impact and move the
marker there before re-checking the grid. Do not skip forward to a louder snare,
tail, or internal body hit after the first clean punch.

## BPM And The One

The BPM is normally in the drums-stem filename, for example
`drums_145_...` means 145 BPM. Use that BPM like a metronome.

Count the phrase as `one, two, three, four` or
`one, two, three, four, five, six, seven, eight`, then back to `one`. The drop
marker must land on that returning `one`. The selected impact should sound and
look like the start of the next bar or phrase, not a middle beat inside it.

The beat clock is a gate, not an excuse to ignore the waveform. The right answer
has both: visible drop-body entry and on-the-one timing.

### Bar-Zero Calibration From A Definitive Drop

When the first drop is messy, fake, or visually ambiguous, use a definitive later
drop to calibrate bar zero. The later anchor can be the second, third, fourth, or
any other clearly hard-hitting section where the drums and bass enter together
on an unmistakable one. Once that later true one is identified, use it to repair
the BPM grid phase, then back-propagate the corrected grid to the first true
drop.

This calibration step is allowed only to fix the metronome phase. It does not
move the final target away from the first true drop when the first drop can be
found. The final marker should still land on the first real drop body, using the
grid phase learned from the definitive later drop.

Reject inferred bar-zero phases that make the visible body entry land on beat
two, beat three, beat four, a snare/tail, or a half-bar-late point. Prefer the
grid phase that aligns the strongest later drop body and the first true drop
body on their returning ones.

## Visual Workflow

1. Start from the whole drums-stem waveform. The drums stem includes the drums
   and bass energy used to identify the drop body.
2. Find the first true drop region by visual structure, not just the first
   transient. Look for a reset, dip, buildup, break, or reduced-energy section
   followed by a sustained hard body.
3. If the first drop is missing, weak, fake, or too ambiguous, use the second
   true drop rather than forcing a bad first-drop marker.
4. If the first drop is visually hard to parse, find a definitive later drop
   first and use it to calibrate bar zero. Then return to the first true drop
   and place the marker on the matching corrected-grid one.
5. Zoom in very far on the chosen drop region.
6. Place the marker at the first sample-level edge where the waveform changes
   into the hard-hitting body: the punch/boom/drop entry, not the tail after it.
7. Confirm the marker still lands on the one using the BPM grid.
8. Confirm there is visible waveform energy directly under the marker line. If
   the marker is on a blank gap, reject it and scan to the first real body
   impact.

## Implementation Plan

The visual detector should implement the transcript as a repeatable pipeline,
not as one-off track fixes:

1. Build the whole-track drums-stem section map first. Candidate scoring may use
   RMS, bass, drums, novelty, and continuity, but the decision has to model the
   visible waveform sections: small intro or buildup, reset or gap, then a
   thicker sustained body.
2. Pick the first definitive thick drop section when it is visually clear. If an
   earlier section is only busy buildup, fake drop, or smaller drums without the
   later bigger body, skip it.
3. Use the BPM from the filename or inferred beatgrid as the metronome source.
   Every selected body entry must be checked against the returning one.
4. When the grid phase is suspect, find a cleaner later drop and use that later
   definitive body entry to repair the metronome phase. Then carry that phase
   back to the first true drop.
5. After the section is chosen, zoom to the first front edge where the waveform
   changes from smaller or quiet material into the hard drum/bass body. The
   marker belongs on that edge, not on a later hit inside the same thick chunk.
   Treat the darkest/fattest sustained waveform region as body evidence: the
   correct marker is the first boundary into that region, not an isolated
   pre-hit or early knee before the dense body starts.
6. Run a second-look verification pass before returning the marker. The second
   look must reject or flag candidates that are blank, off-one, inside the body
   after an earlier edge, one beat or half a bar late, or weaker than a later
   obvious drop. It must also flag markers that lack enough visual waveform
   components to prove the drop, or that sit in dense continuing drums without
   a clear reset/front edge.
7. Add a regression test for each skipped or corrected failure family. Passing
   one specific track is not enough; the detector rule has to explain the visual
   shape that caused the failure.

Current implementation hooks:

- `visual_chunk_candidates(...)` builds the full-waveform visual candidate list.
- `select_first_visual_chunk(...)` chooses the first real visual drop section.
- `_track_zero_grid_phase_guard_candidate(...)` repairs bad BPM grid phase when
  bar zero makes the body land off the one.
- `_blank_waveform_marker_guard_candidate(...)` vetoes empty blue markers and
  moves to the first credible body impact. If the empty marker is a phrase/grid
  edge before the body, the guard must scan far enough to reach the first real
  impact after the gap instead of only checking the next beat.
- `boom_body_section_candidates(...)` builds the whole-track Boom body map from
  sustained darkness, bass/drum simultaneity, drum continuity, contrast, and
  phrase position. It is section evidence, not a standalone license to accept a
  marker.
- Boom body sections must be split on visible internal reset gaps. If a smoothed
  body region contains a thin/quiet bar followed by a renewed darker drum/bass
  body, that re-entry is a new front edge. Smoothing is useful for finding body
  regions, but it must not hide the exact transition where the marker belongs.
- `drop_aligner.boom_profile.marker_boom_proof(...)` is the production proof
  gate. It checks whether the final marker lands on the front edge of a
  dominant Boom body, whether the section matches thresholds learned from human
  reviewed markers, and whether an earlier dominant Boom body should have won.
  File-start Boom sections before the first credible phrase are not allowed to
  disqualify later proven drop bodies unless the marker itself is being checked
  at that opening edge.
- The final selected marker must be normalized after all selector, grid-phase,
  and GUI repairs. Recompute Boom proof and GUI-mask proof from the final marker
  itself before returning, writing candidate JSON, building ALS files, or
  validating. The WebGUI, production builder, and validator must never judge
  different proof payloads for the same marker.
- Proof-backed early-clear markers are final first-drop evidence, not unresolved
  grid candidates. When an early clear drop reclaim passes Boom proof and GUI
  proof, later title-grid or global-phase rescue paths must not pull it forward
  to a nearby later body hit unless that earlier marker itself fails the current
  production contract.
- Early-clear and early-body-peak reclaims must pass the same first-drop body
  contract as any other final marker: on-one timing, sustained post-drop body,
  drum continuity, bass/body simultaneity, and a credible visual reset/front.
  If the early marker fails that contract and a later candidate passes it, the
  early marker is treated as a fake/body-fragment candidate and must yield.
- A close-up GUI nearest-front-edge snap is only a repair for markers that still
  lack usable GUI body/front-edge proof. If the pre-snap marker already passes
  Boom proof, GUI proof, audit, and the first-drop body contract, keep that
  transition boundary; do not move it later to another placeable mask edge
  inside the same body.
- High-resolution analysis failures must recover through the same visual
  detector, not through RMS guessing. If 44.1 kHz visual analysis produces no
  candidate, or produces a final marker that fails audit, Boom proof, GUI
  front-edge proof, or BPM-one proof, the detector may retry lower analysis
  rates such as 16 kHz or 22.05 kHz. A recovered marker is accepted only when it
  independently passes the full production contract; otherwise the track must
  hold. Production builders must not manufacture `visual_first_rms_body_fallback`
  rows.
- `_boom_proven_global_replacement_candidate(...)` is a slow-tempo fail-closed
  repair path. If the selected marker fails Boom proof or deterministic audit,
  it may promote the earliest whole-track dominant Boom front edge, but only
  when that edge passes Boom proof and the normal visual audit after promotion.
- `_same_boom_section_front_edge_replacement(...)` handles the case where the
  detector has landed several bars inside the same sustained Boom body. It may
  collapse that marker back to the section's proven front edge only when the
  current marker is already audit-problematic as off-one or late dense
  continuation, the pullback is several bars into the same section, and the
  section front passes the trained Boom profile. It must not pull normal
  one-bar or two-bar body entries back to coarse section rectangles.
- `visual_boom_on_one_edge_snap` is a narrow cleanup for slow-tempo tracks whose
  marker already passes Boom proof but sits a small off-one distance away from
  the proven Boom front edge. It snaps to the on-one front edge; it does not
  rescue weak, profile-below-threshold, or fallback markers.
- `_visual_chunk_front_edge_before_time(...)` and related body-entry verifiers
  pull late markers back to the front edge of the thick waveform chunk.
- `_selector_locked_micro_boundary_refinement(...)`,
  `_selector_locked_pre_raw_frame_correction(...)`, and
  `_selector_locked_beat_phase_probe_refinement(...)` are close-up refinements
  for high-confidence visual selector candidates. They may repair a raw visual
  frame that is slightly early, late, or beat-phase shifted, but they must not
  displace a selector candidate already accepted as the visible local body entry.
- `audit_visual_selection(...)` is the deterministic second look. It records
  replace/review flags instead of silently trusting the first selected marker.
- Batch rescans must not write visual-first markers whose audit status is
  `review` or `replace` unless an explicit override is used. Full-library
  production builds must also require `boom_proof.passes == true`; missing Boom
  proof is stale detector output and must be held.
- Full-library fresh builds must reject stale/manual-memory final selector
  sources such as `historical_review_memory`, `historical_human_marker`,
  `saved_*`, `web_accept_blue_marker`, `web_save_placed_marker`, and legacy
  `visual_drop_v2` rows. A stale source can only contribute as seed evidence if
  the current visual/Boom/GUI/on-one contract normalizes it to a new proven
  visual source such as `visual_gui_boom_front_edge_contract`.
- The WebGUI waveform and exported review PNGs must use the same global
  boom-body eligibility view as the detector proof. Quiet, flat, or
  below-boom-floor waveform regions must not be drawn as a continuous centerline
  or faint selectable waveform. In visual-first mode, RMS, peak, and sample
  renderers should segment drawing to `boom_relevant_mask` and
  `boom_placeable_mask` so irrelevant bins disappear visually and cannot look
  like valid body evidence.
- Manual click placement in visual-first mode must fail closed. If the current
  drums waveform tile is missing, stale/out of range, non-drum, or lacks enough
  RMS/Boom bins to prove a body after the click, the GUI must reject the click
  instead of creating a green marker. Deep sample inspection should only appear
  in viewports that contain placeable Boom body evidence, so tiny quiet noise
  cannot become visually important just because the user zoomed in.
- Deep sample inspection in the WebGUI must also be masked per sample/bin by
  `boom_relevant_mask`/`boom_placeable_mask`. A tile containing one valid Boom
  edge is not enough to draw raw low-level waveform material across the rest of
  the viewport; irrelevant samples should visually collapse back to the
  centerline.
- `boom_placeable_mask` is a front-edge mask, not a general body mask. The
  sustained body can remain visibly dark after the edge, but placement must only
  be available at the first transition into that body, with a small
  millisecond-level edge grace window for click precision.
- Sparse pulse-train drops may require stable-wide GUI proof. If a narrow
  marker-centered tile contains visible/relevant sparse signal at the marker
  but its nearest placeable bin belongs to an earlier local pulse, the detector
  must check the wider visual context before snapping. When the marker passes
  stable-wide Boom proof, generic nearest-mask repair must not pull the marker
  backward to the earlier pulse; keep the proven on-one sparse body entry or
  accept it through actual-body proof.
- Sparse-pulse repair is a fallback, not the main path for dense Boom bodies.
  Normal GUI/front-edge and dominant Boom-row GUI snap repairs must run first.
- Final candidate-pool promotion may replace a current marker that lacks strict
  GUI front-edge/body proof with a later candidate only when the later candidate
  has fresh GUI actual-body or leading-front-edge proof, Boom proof, audit pass,
  and on-one evidence. Earlier Boom-only preferred rows do not block the
  promotion unless they also prove a strict GUI body/front edge.
- Sparse but strong GUI bodies are valid only when the marker has visible
  signal, immediate body, meaningful post-marker body occupancy, a strong RMS
  peak, nonzero placeable mask, and near-zero placeable-edge offset. This keeps
  lean drops from being rejected while preserving the blank-waveform veto.
- A final on-one snap is allowed only for small millisecond offsets from the
  nearest BPM one, and only when the snapped timestamp independently passes GUI
  actual/leading proof, Boom proof, and visual audit. The snap is not allowed to
  move a marker onto a grid line that lacks visible drop-body evidence.
- GUI front-edge repair should prefer cached visual tile evidence over blind
  probing. When a sparse marker and its raw GUI chunk disagree, inspect
  `boom_body_mask` and `boom_placeable_mask` inside that tile, choose the last
  placeable edge immediately before the first sustained body run, then re-check
  Boom proof, GUI proof, audit, and BPM-one. This models the human zoom-in
  workflow without adding per-track constants.
- A narrow GUI tile can show placeable sparse signal while missing immediate
  body density. If Boom proof verifies the body under that marker, preserve the
  distinction as stable-wide sparse-front proof instead of treating it as a
  generic nearest-mask snap. Stable-wide sparse proof must still require visible
  signal at the marker and must not bypass the blank-waveform veto.
  A sparse pulse must yield when a nearby stronger/darker Boom body front edge
  exists, or when it sits inside the same sustained body after an earlier
  comparable front edge without a fresh reset/re-entry.
- Sparse-pulse fallback must be bounded. Pre-rank cheap visual sparse-pulse
  candidates first, then run expensive wide GUI-mask proof only on a short
  strongest shortlist so one track cannot hang a full-library audit.
- Sparse/staccato body fronts may pass the strict GUI mask only when the marker
  still has visible signal and the candidate proves a strong immediate visual
  body from track-relative profile, body score, darkness, post-body energy, drum
  continuity, bass, and simultaneity. This relief is for real staccato drop
  fronts and continuous hard-body waveforms whose body is visible in the wider
  waveform even when fine density undercounts the immediate body; it is not
  permission to accept blank marker lines or arbitrary sparse intro ticks.
- Nearby title-grid restoration is a close-up correction, not a whole-track
  replacement strategy. Use it only after the detector has already chosen a
  credible body neighborhood, then zoom from a nearby on-one seed to the first
  visible front edge. Do not run this restore in the early intro zone or let it
  preserve a weak/sparse early marker when a far-later full-track body is
  materially stronger in profile, bass, contrast, and score.
- If a sparse or intro-like marker survives late in the pipeline, run one final
  visual sanity check against the current candidate list. A far-later
  overwhelming full-track body can reclaim the marker only when it is already a
  proven detector candidate and is clearly stronger on the body metrics that
  define a drop. Prefer the already corrected visible-signal/front-edge version
  of that body over a raw full-track edge inside the same region.
- Dominant Boom body rows may be snapped to their nearest GUI placeable front
  edge before sparse fallback. The snapped marker still has to pass Boom proof,
  GUI proof, visual audit, and BPM-one proof; this is a way to recover the first
  visible hard-body edge, not permission to choose a later body hit.
- Coarse GUI proof is not enough for final acceptance. A marker that passes in
  a wide tile must also survive a close-up front-edge proof. If the close-up
  proof shows the marker is inside an already-started body, on a blank pre-edge,
  or on a neighboring pulse, repair to the first nearby close-up
  `boom_placeable_mask` edge instead of accepting the coarse pass. Local fine
  repair should try the close-up GUI's nearest placeable front edge before
  accepting the seed marker, then fall back to the seed only when the nearest
  edge fails Boom proof, GUI proof, audit, or BPM-one proof.
- A passing GUI mask is still not production clean unless the marker has
  relevant signal, is on the placeable front-edge mask, and has immediate
  post-marker Boom body occupancy. Relevant waveform activity alone is not
  enough: if `marker_immediate_body_present` is false and the 250/500 ms
  post-marker body occupancy is near zero, the detector must fail closed or use
  an explicit sparse-front proof path. Do not let a single isolated spike or
  non-body relevant mask rescue a marker that lacks sustained body immediately
  after the line.
- Context candidates can overrule stale Boom-section edges. When final proof
  fails because the selected marker is blank/weak, scan the already-generated
  visual/body candidates for a zoomed GUI-proven front edge before promoting an
  earlier dominant Boom section. A bar-level Boom edge that is visually blank in
  the close-up view must never replace a candidate whose own close-up GUI proof,
  Boom proof, audit, and BPM-one proof pass.
- First clear drop beats later darker sections. If final output has moved many
  seconds to a later strong/dark body, the detector must give early visual/body
  candidates a bounded close-up GUI repair pass first. An early candidate that
  passes close-up GUI proof, Boom proof, audit, and BPM-one proof must be kept
  as the first true drop instead of being overwritten by a later darker section.
- Dense continuation is not a drop front. If the current marker sits in drums
  that were already busy before the line, and a later candidate has a much
  cleaner reset plus stronger profile, body score, darkness, bass, drum
  continuity, and contrast, the later clean reset may replace the current marker
  only after it passes Boom proof, GUI proof, audit, and BPM-one proof.
- Same-region weak fronts may yield to the next on-one hard body. If a first
  credible/front-edge repair lands on a weaker lead-in body and a nearby
  same-region candidate is on the returning one with stronger sustained body,
  post-body energy, bass, and drum continuity, promote the stronger front only
  after fresh Boom proof, GUI proof, audit, and BPM-one checks pass.
- A weak/off-phrase early body may yield to an overwhelming later clean body
  only when the later body is materially stronger across the full visual
  evidence stack. This is a strict second-drop fallback, not a general "pick the
  loudest section" rule.
- Later-repeat pullback must be reversible. If a guard pulled a later definitive
  repeat back to an earlier body, keep the earlier body only when it is truly
  comparable. If the later repeat is clearly darker, stronger, cleaner, and more
  reset-like, restore the later definitive front edge.
- A GUI-failed early marker is weak evidence even when older Boom metrics look
  strong. If the marker fails the placeable/front-edge mask, the whole-track GUI
  scan may use the first later impact-leading placeable edge, but that edge must
  independently pass Boom proof, GUI proof, audit, and BPM-one proof.
- Proven grid-one snaps are not thin-marker fallbacks. A
  `visual_boom_grid_one_snap` marker that is on-one and already has Boom proof
  or a tight Boom nearest offset must not be moved many seconds later merely
  because a later body has heavier GUI occupancy.
- Proven Boom markers may use stable-wide relief only after normal close-up
  repair fails. This relief is limited to full-track stronger and earliest
  dominant Boom replacements that already pass Boom proof, audit, BPM-one proof,
  visible marker signal, and coarse GUI proof. It exists for fragmented pulse
  masks where the close-up mask disagrees with the stable wide view; it must not
  apply to blank markers, sparse fallbacks, stale memory, or context candidates
  that can be repaired to a real close-up front edge.
- WebGUI dark chunk/highlight rectangles must be front-edge locked. A body-only
  tail may render as simplified boom-body bars for visual context, but it must
  not render as a selectable-looking rectangular chunk unless the chunk contains
  `boom_placeable_mask` evidence, and the rectangle's left edge must be pinned
  to the first placeable bin rather than to the wider smoothed body mask.
- GUI front-edge repair must prefer sustained placeable runs over isolated mask
  speckles. A tiny one-bin mask point near the current marker is not enough to
  override the first sustained hard-body front edge in the same local view.
- GUI repair stabilization is local-only. After the first repair chooses a
  region, the stabilizing pass may refine that same neighborhood, but it must
  not use whole-track fallback to jump to an intro, a later section, or another
  drop.
- Whole-track GUI replacement is a fail-closed rescue for markers that are
  off-grid, blank, or otherwise not a proven visual body. It must not relocate a
  trusted on-one selector or GUI body marker by many seconds just because another
  section has a darker mask. If such a cross-section GUI repair occurs, restore
  the strong previous on-one body candidate and log the rejected repair.
- Whole-track GUI repair must not depend only on smoothed body density. A
  placeable impact that leads into a sustained body can be a real drop front
  even when the long-window density percentile is low. Such impact-leading rows
  may be proposed only from `boom_placeable_mask` evidence, and they still must
  pass Boom proof, GUI proof, visual audit, and BPM-one proof before acceptance.
- If a final marker is only a few milliseconds away from the raw BPM-derived
  returning one and fails the GUI front-edge mask at its calibrated timestamp,
  the detector may snap to the nearby raw one. This is allowed only when the raw
  one itself passes Boom proof, GUI proof, visual audit, and BPM-one proof.
- Audit-preferred alternatives are not automatically authoritative. If final
  GUI repair finds a later front edge that passes Boom proof, GUI front-edge
  proof, and BPM-one proof, an earlier audit-preferred body can block that
  repair only when the earlier body also passes the same production contract.
  Preferred bodies that fail the GUI placeable-mask proof are evidence to
  investigate, not a license to keep a stale marker.
- BPM phase calibration for GUI/front-edge repairs is metadata repair, not
  section selection. Real selected detector candidates may calibrate small
  visual front-edge offsets so `one_distance_ms` reflects the repaired marker;
  anonymous track-grid checks should remain on the raw BPM grid unless the
  offset is outside the normal one tolerance.
- The WebGUI server must enforce the same rule on review writes. In
  visual-first mode, `/api/approve` and `/api/correct` must reject a marker
  before logging or regenerating ALS unless the marker passes Boom proof, sits
  on the BPM one, and is within the tight front-edge tolerance of the proven
  Boom body. Client-side click rejection is not enough.
- The WebGUI save guard, fresh production builder, and production validator
  must use the same visual-first production analysis rate. If one path silently
  drops back to a lower-resolution feature map, the displayed marker, saved ALS,
  and validation report are no longer proving the same Boom front edge.
- Boom proof may only rescue small GUI-mask alignment errors when the local GUI
  tile still contains placeable Boom front-edge evidence. It must not turn an
  all-empty or all-rejected GUI tile into a passing proof, because that makes a
  visually unavailable region selectable by metadata alone.
- `visual_first_marker(...)` must fail closed when its final marker cannot pass
  audit, Boom proof, GUI placeable-mask proof, and BPM-one proof. If recovery
  cannot produce a fully proven replacement, the detector should return an
  explicit hold/error instead of `ok: true`.
- Green/manual marker validation must be marker-only. A posted or nearby
  candidate may be useful for display, but it must not be passed into the
  visual-first save guard for `SAVE PLACED`, `/api/correct`, or manual
  `/api/regenerate_als`; otherwise stale detector metrics can make a manual
  marker look more proven than the waveform under the marker actually is.
  Candidate context is allowed only for blue approval, where the exact current
  detector marker is being accepted and `/api/approve` first checks that the
  browser marker matches the server's current blue marker.
- ALS regeneration is also a write path. `ReviewApp.regenerate_als(...)`,
  including `/api/regenerate_als` and batch utilities that call it, must run the
  visual-first Boom proof/on-one/front-edge guard before backing up or writing
  any ALS file.
- Summary synchronization is also a marker write path. Utilities that copy
  detector-prepared candidate JSON back into summary CSV rows must require
  `visual_audit.status == pass`, no audit flags, `boom_proof.passes == true`,
  and `gui_mask_proof.passes == true` for visual detector payloads. A bare
  detector timestamp is not enough evidence to update the production summary.
- Legacy detector write paths must be locked by default. `batch.py`, `main.py`,
  `auto_review.py`, `apply_auto_place_initial.py`, and
  `reanalyze_remaining_hard_cases.py` are old non-visual writers; they must not
  mutate production summary, candidate JSON, logs, or ALS files unless an
  explicit legacy experiment opt-in is supplied. Normal marker production must
  go through the visual-first Boom-waveform builder and WebGUI guard.
- Visual-first display must fail closed. If the fresh visual-first scan cannot
  produce a usable Boom marker for the current item, the GUI must hide stale
  detector output and expose the track as a `visual_first_hold` with no blue
  marker to accept. Historical memory, old visual candidates, and previous batch
  markers may be retained as diagnostic stale fields, but they must not be drawn
  or accepted as the active marker.
- `train_boom_profile.py` recalibrates `models/boom_profile.json` from
  human-reviewed correction logs. It must use human review sources only, not
  batch-auto detector output.
- Historical manual review memory may correct fresh detector output only through
  the validated human override path. The old marker must come from a manual
  review source and must pass the current Boom proof, front-edge freshness
  check, and GUI Boom-mask proof before it can replace a fresh marker. Stale
  historical rows that no longer pass those current checks are diagnostics, not
  active markers.
- `audit_visual_first_human_review_memory.py` is the conflict gate between a
  fresh production report and historical human review logs. Production reports
  must have zero validated hard mismatches. If a manual marker still passes the
  current waveform proof and disagrees with the report marker, the builder must
  promote it as `visual_validated_human_review_override` or hold the build.
- `validate_visual_first_production.py` is the final all-library gate. A set is
  not production-ready until this validator reports every non-excluded row as
  passed. Its failure families are the implementation backlog.
- `audit_visual_first_combined_als.py` is the standalone final-set gate. It
  must pass against the written combined ALS and the exact processed marker
  report, proving every row has complete CH1/CH2/CH3 clips, file references
  match the expected stems, and each clip has the BeatTime `0` anchor at the
  validated marker within tolerance. This protects against a correct detector
  report producing a bad playable Ableton set.
- `audit_visual_first_suspicious_markers.py` is the independent suspicion gate
  after production validation. It recomputes local GUI proof for every saved
  marker, requires fresh persisted Boom proof, and writes a full-track advisory
  CSV for earlier/later much-stronger Boom sections. Production artifacts must
  have zero hard suspicious-marker failures. Advisory rows are not automatic
  failures because a later louder section can still be a valid second drop while
  the first true drop remains the right target, but repeated advisory families
  should become regression fixtures when user review confirms them.

Blue/green review semantics:

- Blue is the detector marker. Accepting it logs `web_accept_blue_marker`.
  In visual-first mode, blue acceptance must use the dedicated approve path for
  the current Boom-proof detector marker, not the generic manual/candidate
  correction path.
- Green is the manually placed `1.1.1` marker. `SAVE PLACED` logs the manual
  correction separately. In visual-first mode it remains a green/manual marker
  even if the click is near a detector candidate; the save guard must prove the
  marker from the waveform and grid instead of inheriting candidate proof.
- The old candidate-number, knee, refine, and scan accept paths should not be
  used for normal visual-first review.
- The old refine-marker and non-visual auto-place API modes are disabled while
  the WebGUI is in visual-first mode. The only active choices are accepting the
  Boom-proof blue marker, placing/saving a Boom-gated green marker, or skipping.
- `RETRAIN NOW`, section-label logging, and legacy auto-batch fallback are also
  disabled in visual-first WebGUI mode. Visual-first production review must start
  from an explicit visual-first summary/build artifact; it must not silently run
  old `batch.py` or mutate old ranker training state from inside the review UI.
- `/api/correct` in visual-first mode is green/manual only. It must reject
  `web_accept_blue_marker` requests and ignore candidate-pick metadata so direct
  API calls cannot pollute human review memory with fake blue or old
  candidate-pick semantics.

## What Counts As A Drop

A true drop is the transition into sustained drum/bass impact. It usually has
stronger low end, denser drum continuity, and a clear body after the boundary.
The correct edge is often just after pre-drop space, a riser, a fill, or a reset.

Do not confuse these with drops:

- Intro drums or early low-bass loops.
- Buildup hits, pre-hits, fills, or one-off impacts before the body.
- Vocal or instrumental texture changes without the hard drum/bass body.
- Fake drops when a later reset leads to the real sustained body.
- Dense opening material unless the track truly starts at the drop body.
- Any candidate that lands on beat two, beat three, beat four, or off-grid.
- Any candidate that lands on silence or a near-empty waveform gap.

## Training And Regression Rules

Manual saves and skipped examples are training signals. When tracks are skipped
because the marker is not on a drop or not on the one, treat that as detector
failure evidence. Add representative skipped/corrected cases as regression
fixtures and prefer rule changes that explain the whole failure family.

Treat large MicroSnap moves as suspect. A snap nearly a beat away from the
visual edge is not proof of a drop; it usually means the chosen section was
wrong or the snap found body/tail material.

Do not push detector changes until the review set and regression tests pass and
the user explicitly asks for a push.

## Current Named Failure Families

- Early low-bass intro selected while a later reset-and-body drop is stronger.
  Use the late reset/body guard.
- Short pre-hit before the body. Zoom in and place just before the sustained
  waveform transition, not on the tail.
- Dense opening tracks. If the first bars are already sustained
  drums/bass/full-spectrum energy, use the opening-drop path only when it is
  truly the drop body and on the one.
- Texture-heavy buildup. If instrumental/vocal texture drops away and
  drums/bass take over, the body entry is the drop.
- Off-one candidates. If the visual body starts near a candidate but the chosen
  point is beat two, beat three, beat four, or off-grid, repair the grid choice
  instead of approving it.
- Ambiguous bar zero. If a later definitive drop clearly proves the grid phase,
  use that later drop to calibrate bar zero, then back-propagate the corrected
  one to the first true drop.
- Blank marker on a grid edge. If the grid edge is visually empty, use the blank
  waveform guard: reject the empty blue marker and choose the first credible
  confirmed body impact inside the same drop region. A phrase-start line with
  little or no waveform under it is not the drop just because the following bar
  average is high; keep scanning to the first visible hard body entry.
- Early off-phrase sustained block. If the blue marker lands inside an early
  dense/fake block before the definitive low/body section, use the later
  phrase-start drop as the anchor and choose the first visual entry of that
  later section, then zoom to the first punch.
- Nearby intro/fill before phrase body. If an early candidate is a busy buildup
  or fill and the next phrase-start section has sustained drum continuity,
  stronger low end, and a clear instrumental/vocal texture release, select that
  phrase body entry instead of protecting the earlier fill.
- Backward MicroSnap before phrase body. If the visual phrase body entry is
  already on the BPM one, do not let MicroSnap pull the marker backward into
  the pre-drop tail; keep the marker on the visual body-entry edge.
- Small forward MicroSnap into the tail. If the visual edge is already a clean
  local re-entry on the one, do not let MicroSnap move the marker forward into
  the first hit's decay/tail unless the snap is clearly higher-confidence than
  the visual edge. The blue line belongs on the first body edge, not where the
  hit starts fading.
- Early knee before dark body. If the close-up marker lands on a weak pre-hit or
  knee and the waveform becomes a sustained dark/fat drum body shortly after it,
  move the marker to that body-entry boundary. A thin early hit is not the drop
  when the real dense waveform body begins after it.
- Late marker inside the same thick chunk. If the selected blue marker is inside
  a large sustained waveform block and there is a clear quiet-to-thick front edge
  shortly before it, move back to that first front edge. Do not mark the later
  boom/decay inside the same chunk as the drop.
- Beat-phase probe overriding a correct visible body entry. If the visual
  selector accepts the raw chunk as a visible local body-entry, do not let a
  later half-beat or beat-fraction probe move the marker into the next hit or
  beat. Beat-phase probes are for repairing visually shifted candidates, not
  replacing an already correct raw body entry.
- Missing or ambiguous visual evidence. Structure, fusion, and heuristic
  candidates must still provide visual waveform proof. If the selected marker has
  weak sustained drum body, high pre-drop drum continuity with no clear reset, or
  no visual components for the audit to inspect, hold it for review instead of
  auto-accepting it.
