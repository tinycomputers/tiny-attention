// Scene timing. Each scene lasts at least its visual minimum, and long enough for its narration plus a tail.
import { FPS } from "./theme";
import vo from "./vo.json";

export type SceneId = "hook" | "intro" | "tiles" | "engine" | "memory" | "trust" | "cta";
const order: SceneId[] = ["hook", "intro", "tiles", "engine", "memory", "trust", "cta"];
const minSec: Record<SceneId, number> = { hook: 4.5, intro: 6.5, tiles: 10.4, engine: 8.6, memory: 9.2, trust: 7.8, cta: 9.0 };
const voDelay: Record<SceneId, number> = { hook: 0.4, intro: 0.5, tiles: 0.5, engine: 0.5, memory: 0.5, trust: 0.45, cta: 0.6 };
const tail = 0.7;

export const SCENES = {} as Record<SceneId, { from: number; dur: number; voFrom: number; voDur: number; text: string }>;
let fr = 0;
for (const id of order) {
  const v = (vo as Record<string, { text: string; sec: number }>)[id];
  const dur = Math.round(Math.max(minSec[id], voDelay[id] + v.sec + tail) * FPS);
  SCENES[id] = { from: fr, dur, voFrom: fr + Math.round(voDelay[id] * FPS), voDur: Math.round(v.sec * FPS), text: v.text };
  fr += dur;
}
export const TOTAL = fr;
