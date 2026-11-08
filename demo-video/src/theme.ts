// One theme for the whole video. Colours are the app's own tokens (web/src/styles/global.css, dark theme).
import { Easing, spring, SpringConfig } from "remotion";

export const C = {
  bg: "#0d1117",
  bgDeep: "#070a0f",
  panel: "#161b22",
  panelHi: "#1c2330",
  line: "#30363d",
  fg: "#c9d1d9",
  white: "#f0f6fc",
  muted: "#8b949e",
  accent: "#58a6ff", // hero colour
  accentDim: "#1f6feb",
  ok: "#3fb950",
  err: "#f85149",
} as const;

export const FONT = {
  sans: '"Inter Variable", system-ui, sans-serif',
  mono: '"JetBrains Mono Variable", ui-monospace, monospace',
} as const;

export const FPS = 30;
export const W = 1920;
export const H = 1080;

export const EASE = {
  out: Easing.bezier(0.16, 1, 0.3, 1),
  inOut: Easing.bezier(0.65, 0, 0.35, 1),
  in: Easing.bezier(0.7, 0, 0.84, 0),
  back: Easing.bezier(0.34, 1.56, 0.64, 1),
};

export const SPR: Record<string, Partial<SpringConfig>> = {
  smooth: { damping: 26, stiffness: 120, mass: 0.9 },
  snappy: { damping: 20, stiffness: 190, mass: 0.8 },
  pop: { damping: 13, stiffness: 170, mass: 0.7 },
  soft: { damping: 32, stiffness: 70, mass: 1.1 },
};

export const sp = (frame: number, fps: number, delay = 0, cfg: keyof typeof SPR = "smooth") =>
  spring({ frame: frame - delay, fps, config: SPR[cfg] });

export const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
