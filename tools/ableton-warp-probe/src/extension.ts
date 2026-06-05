import {
  AudioClip,
  AudioTrack,
  ClipSlot,
  Clip,
  DataModelObject,
  initialize,
  WarpMode,
  type ActivationContext,
  type ArrangementSelection,
  type ClipSlotSelection,
  type ExtensionContext,
  type Handle,
} from "@ableton-extensions/sdk";

import * as crypto from "node:crypto";
import * as fs from "node:fs/promises";
import * as path from "node:path";

type Api = ExtensionContext<"1.0.0">;

interface ExportPayload {
  schema: "track_organizer_ableton_warp_markers_v1";
  exportedAt: string;
  audioPath: string;
  clipName: string;
  songTempo: number | null;
  clip: {
    startTime: number;
    endTime: number;
    startMarker: number;
    endMarker: number;
    looping: boolean;
    loopStart: number;
    loopEnd: number;
    warping: boolean;
    warpMode: number;
    warpModeName: string;
  };
  warpMarkers: Array<{
    sampleTime: number;
    beatTime: number;
  }>;
}

function warpModeName(mode: number): string {
  return WarpMode[mode as WarpMode] ?? String(mode);
}

function stableName(audioPath: string): string {
  return `${crypto.createHash("sha1").update(audioPath).digest("hex").slice(0, 20)}.json`;
}

async function writeIfPossible(filePath: string, text: string): Promise<string | null> {
  try {
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, text, "utf8");
    return filePath;
  } catch (error) {
    console.error(`Could not write ${filePath}:`, error);
    return null;
  }
}

async function writePayload(context: Api, payload: ExportPayload): Promise<void> {
  const text = `${JSON.stringify(payload, null, 2)}\n`;
  const written: string[] = [];

  const sidecar = await writeIfPossible(`${payload.audioPath}.ableton_warp_markers.json`, text);
  if (sidecar) written.push(sidecar);

  const storageRoot = context.environment.storageDirectory ?? context.environment.tempDirectory;
  if (storageRoot) {
    const exported = await writeIfPossible(
      path.join(storageRoot, "track-organizer-warp-markers", stableName(payload.audioPath)),
      text,
    );
    if (exported) written.push(exported);
  }

  if (!written.length) {
    throw new Error(`Could not write warp marker export for ${payload.audioPath}`);
  }
  console.log(`TrackOrganizer warp marker export: ${written.join(", ")}`);
}

function payloadForClip(context: Api, clip: AudioClip<"1.0.0">): ExportPayload {
  return {
    schema: "track_organizer_ableton_warp_markers_v1",
    exportedAt: new Date().toISOString(),
    audioPath: clip.filePath,
    clipName: clip.name,
    songTempo: context.application.song?.tempo ?? null,
    clip: {
      startTime: clip.startTime,
      endTime: clip.endTime,
      startMarker: clip.startMarker,
      endMarker: clip.endMarker,
      looping: clip.looping,
      loopStart: clip.loopStart,
      loopEnd: clip.loopEnd,
      warping: clip.warping,
      warpMode: clip.warpMode,
      warpModeName: warpModeName(clip.warpMode),
    },
    warpMarkers: clip.warpMarkers.map((marker) => ({
      sampleTime: marker.sampleTime,
      beatTime: marker.beatTime,
    })),
  };
}

async function exportClip(context: Api, clip: AudioClip<"1.0.0">): Promise<void> {
  await writePayload(context, payloadForClip(context, clip));
}

function clipOverlapsSelection(clip: Clip<"1.0.0">, selection: ArrangementSelection): boolean {
  return clip.endTime > selection.time_selection_start && clip.startTime < selection.time_selection_end;
}

async function exportTrack(context: Api, track: AudioTrack<"1.0.0">): Promise<number> {
  let exported = 0;
  for (const clip of track.arrangementClips) {
    if (clip instanceof AudioClip) {
      await exportClip(context, clip);
      exported += 1;
    }
  }
  for (const slot of track.clipSlots) {
    const clip = slot.clip;
    if (clip instanceof AudioClip) {
      await exportClip(context, clip);
      exported += 1;
    }
  }
  return exported;
}

async function exportSong(context: Api): Promise<number> {
  let exported = 0;
  for (const track of context.application.song.tracks) {
    if (track instanceof AudioTrack) {
      exported += await exportTrack(context, track);
    }
  }
  return exported;
}

export function activate(activation: ActivationContext) {
  const context = initialize(activation, "1.0.0");

  context.commands.registerCommand("trackOrganizer.exportClipWarpMarkers", (arg: unknown) => {
    void (async () => {
      const clip = context.getObjectFromHandle(arg as Handle, Clip);
      if (!(clip instanceof AudioClip)) {
        console.error("TrackOrganizer export requires an audio clip.");
        return;
      }
      await exportClip(context, clip);
    })().catch((error) => console.error(error));
  });

  context.commands.registerCommand("trackOrganizer.exportSelectionWarpMarkers", (arg: unknown) => {
    void (async (selection: ArrangementSelection) => {
      const tracks = selection.selected_lanes
        .map((handle) => context.getObjectFromHandle(handle, DataModelObject))
        .filter((obj): obj is AudioTrack<"1.0.0"> => obj instanceof AudioTrack);

      let exported = 0;
      for (const track of tracks) {
        for (const clip of track.arrangementClips) {
          if (clip instanceof AudioClip && clipOverlapsSelection(clip, selection)) {
            await exportClip(context, clip);
            exported += 1;
          }
        }
      }
      console.log(`TrackOrganizer exported ${exported} audio clip warp marker file(s).`);
    })(arg as ArrangementSelection).catch((error) => console.error(error));
  });

  context.commands.registerCommand("trackOrganizer.exportTrackWarpMarkers", (arg: unknown) => {
    void (async () => {
      const track = context.getObjectFromHandle(arg as Handle, DataModelObject);
      if (!(track instanceof AudioTrack)) {
        console.error("TrackOrganizer track export requires an audio track.");
        return;
      }
      const exported = await exportTrack(context, track);
      console.log(`TrackOrganizer exported ${exported} audio clip warp marker file(s) from ${track.name}.`);
    })().catch((error) => console.error(error));
  });

  context.commands.registerCommand("trackOrganizer.exportSlotWarpMarkers", (arg: unknown) => {
    void (async () => {
      const slot = context.getObjectFromHandle(arg as Handle, ClipSlot);
      const clip = slot.clip;
      if (!(clip instanceof AudioClip)) {
        console.error("TrackOrganizer slot export requires an audio clip slot.");
        return;
      }
      await exportClip(context, clip);
      console.log("TrackOrganizer exported 1 session clip warp marker file.");
    })().catch((error) => console.error(error));
  });

  context.commands.registerCommand("trackOrganizer.exportSelectedSlotWarpMarkers", (arg: unknown) => {
    void (async (selection: ClipSlotSelection) => {
      let exported = 0;
      for (const handle of selection.selected_clip_slots) {
        const slot = context.getObjectFromHandle(handle, ClipSlot);
        const clip = slot.clip;
        if (clip instanceof AudioClip) {
          await exportClip(context, clip);
          exported += 1;
        }
      }
      console.log(`TrackOrganizer exported ${exported} selected session clip warp marker file(s).`);
    })(arg as ClipSlotSelection).catch((error) => console.error(error));
  });

  context.commands.registerCommand("trackOrganizer.exportSongWarpMarkers", () => {
    void (async () => {
      const exported = await exportSong(context);
      console.log(`TrackOrganizer exported ${exported} audio clip warp marker file(s) from the Set.`);
    })().catch((error) => console.error(error));
  });

  context.ui.registerContextMenuAction(
    "AudioClip",
    "TrackOrganizer: Export Warp Markers",
    "trackOrganizer.exportClipWarpMarkers",
  );

  context.ui.registerContextMenuAction(
    "AudioTrack.ArrangementSelection",
    "TrackOrganizer: Export Selection Warp Markers",
    "trackOrganizer.exportSelectionWarpMarkers",
  );

  context.ui.registerContextMenuAction(
    "ClipSlot",
    "TrackOrganizer: Export Slot Warp Markers",
    "trackOrganizer.exportSlotWarpMarkers",
  );

  context.ui.registerContextMenuAction(
    "ClipSlotSelection",
    "TrackOrganizer: Export Selected Session Warp Markers",
    "trackOrganizer.exportSelectedSlotWarpMarkers",
  );

  context.ui.registerContextMenuAction(
    "AudioTrack",
    "TrackOrganizer: Export Track Warp Markers",
    "trackOrganizer.exportTrackWarpMarkers",
  );

  context.ui.registerContextMenuAction(
    "AudioTrack",
    "TrackOrganizer: Export Set Warp Markers",
    "trackOrganizer.exportSongWarpMarkers",
  );
}
