# Visual Drop Detection Rules

The visual-first detector must behave like the review GUI workflow:

1. Look at the whole drum-stem waveform first. The target is the first true drop section, not the first visible transient.
2. A drop is the transition into a sustained drum/bass body. The marker belongs at the beginning of that transition, then micro-aligned in the zoomed view.
3. Skip early intro, buildup, and fake-drop sections when a later section has clearly stronger sustained drum/bass energy after a reset or dip.
4. Use the BPM from the file name or inferred beatgrid to stay near the one, but do not let the clock override the visible transition edge.
5. Treat large MicroSnap moves as suspect. A snap nearly a beat away from the visual edge is not proof of a drop; it usually means the chosen section was wrong or the snap found body/tail material.
6. Compare sections by shape, not only height: intro texture can be loud, but the real drop usually has a sustained body, stronger low end, denser drum continuity, and a visible reset before it.
7. Manual saves and skipped examples are training signals. Add each corrected track as a regression fixture and prefer rules that explain the whole family of failures.
8. Do not push detector changes until the review set and regression tests pass and the user explicitly asks for a push.

Current named failure families:

- Early low-bass intro selected while a later reset-and-body drop is stronger. Use the late reset/body guard.
- Short pre-hit before the body. Zoom in and place just before the sustained waveform transition, not on the tail.
- Dense opening tracks. If the first bars are already sustained drums/bass/full-spectrum energy, use the opening-drop path.
- Texture-heavy buildup. If instrumental/vocal texture drops away and drums/bass take over, the body entry is the drop.
