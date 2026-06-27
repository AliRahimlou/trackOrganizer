# Agent Instructions

This repo is about finding DJ-usable Ableton drop markers. The goal is not to
pick a loud transient or a convenient candidate. The goal is to land `1.1.1` on
the true drop impact, on the musical one, with millisecond-level placement.

## Drop Review Contract

Every accepted marker must pass both checks:

1. Visual drop evidence: the marker is at the start of the hard-hitting,
   sustained drum/bass body in the drums stem waveform.
2. Musical grid evidence: the marker is on the one according to the BPM in the
   track title or the inferred beatgrid.

If either check fails, do not approve the marker.

Blank waveform veto: the blue marker must never sit on silence or a near-empty
gap. If there is no visible waveform energy under the marker line, the marker is
wrong even if the inferred BPM grid says it is a bar start. Move to the first
credible hard body impact in the visible drop region, not the loudest later
snare/body hit inside the same phrase, then re-check the grid phase from that
impact.

The BPM is normally embedded in filenames such as `drums_145_...`; use that as
the metronome source. Count the grid like a DJ: `one, two, three, four` or
`one, two, three, four, five, six, seven, eight`, then back to `one`. Drop
markers belong on that returning `one`, not beat two, beat three, beat four, or
an off-grid body/tail transient.

When bar zero is ambiguous, calibrate the grid from a definitive later drop.
Find the cleanest obvious drop body anywhere in the track, including the second,
third, or later drop if needed. Set the metronome/`1.1.1` relationship from that
known true one, then carry the corrected grid backward to the first true drop.
Do not trust an inferred bar-zero phase when it makes the visible drop body land
on beat two, beat three, a snare/tail, or a half-bar-late point.

## Visual Workflow

Always use visual-first review mode. The review UI must open on the full
drums-stem waveform and work from visible waveform structure before accepting
any marker. Do not use the old AI-window-first review flow for normal review or
detector debugging.

Use the full drums-stem waveform first. The drums stem contains the drums and
bass information needed to identify the real impact section.

Find the first true drop if it is visually clear. If the first drop is missing,
weak, fake, or too ambiguous, use the second true drop rather than forcing a bad
first-drop marker.

If a later definitive drop is easier to identify than the first drop, use that
later drop to fix the musical grid first. After the grid is corrected, return to
the first true drop and place `1.1.1` on its matching on-one body entry.

After locating the likely drop region, zoom in very far and place the marker at
the first sample-level boundary where the waveform changes into the actual
hard-hitting drop body. Look for the big punch/body entry after the pre-drop
space, reset, buildup, or break. Do not place markers on intro hits, buildup
hits, vocal texture, weak pre-hits, tails, or dense sections that are not the
drop.

Before accepting a blue marker, inspect the exact marker line. If the line is on
empty waveform space, reject it and scan inside the same drop region for the
first credible drum/bass body impact.

Review UI semantics: blue is the detector marker and green is the manual placed
marker. Accepted detector markers should be logged as `web_accept_blue_marker`.
Manual `PLACE 1.1.1` plus `SAVE PLACED` corrections remain green/manual review
data.

Skipped tracks are important training evidence. When many tracks are skipped
because the detector is not on a drop or not on the one, treat that as a model
and rules failure, add regression coverage, and tighten the detector before
trusting more auto-saves.

## Related Docs

The detailed detector rules live in
[`docs/visual_drop_detection_rules.md`](docs/visual_drop_detection_rules.md).
Keep code, tests, and review behavior aligned with that document.
