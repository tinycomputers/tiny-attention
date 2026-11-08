import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { C, EASE, FONT, clamp, sp } from "../theme";

/** Entrance: spring on opacity + translate + scale, with a per-item delay. Optional exit. */
export const Enter: React.FC<{ delay?: number; y?: number; x?: number; s?: number; exitAt?: number; children: React.ReactNode; style?: React.CSSProperties; cfg?: "smooth" | "snappy" | "pop" | "soft" }> = ({ delay = 0, y = 28, x = 0, s = 0.97, exitAt, children, style, cfg = "smooth" }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = sp(f, fps, delay, cfg);
  const ex = exitAt === undefined ? 0 : interpolate(f, [exitAt, exitAt + 10], [0, 1], { ...clamp, easing: EASE.in });
  const o = interpolate(a, [0, 1], [0, 1], clamp) * (1 - ex);
  return <div style={{ opacity: o, transform: `translate(${(1 - a) * x + ex * -x * 0.3}px, ${(1 - a) * y - ex * 14}px) scale(${s + (1 - s) * a - ex * 0.02})`, ...style }}>{children}</div>;
};

/** Brand mark: the 4x4 causal tile grid, diagonal in the accent colour. */
export const Mark: React.FC<{ size: number; progress?: number; glow?: number }> = ({ size, progress = 1 }) => {
  const c = size / 4, g = size * 0.045;
  const cells: React.ReactNode[] = [];
  for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) {
    const on = k <= r; const diag = k === r;
    const idx = r * 4 + k;
    const t = interpolate(progress, [idx / 22, idx / 22 + 0.28], [0, 1], { ...clamp, easing: EASE.back });
    cells.push(<rect key={idx} x={k * c + g / 2 + (c - g) * (1 - t) / 2} y={r * c + g / 2 + (c - g) * (1 - t) / 2} width={(c - g) * t} height={(c - g) * t} rx={size * 0.03}
      fill={diag ? C.accent : on ? C.ok : "transparent"} stroke={on ? "none" : C.line} strokeWidth={size * 0.012} opacity={on ? (diag ? 1 : 0.85) : 1} />);
  }
  return <svg width={size} height={size}>{cells}</svg>;
};

/** Small mono section label: index, hairline rule, name. */
export const Eyebrow: React.FC<{ n?: string; children: React.ReactNode }> = ({ n, children }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 16, fontFamily: FONT.mono, fontSize: 19, letterSpacing: "0.12em", textTransform: "uppercase", color: C.muted }}>
    {n && <span style={{ color: C.accent, fontWeight: 600 }}>{n}</span>}
    {n && <span style={{ width: 40, height: 1, background: C.line }} />}
    <span>{children}</span>
  </div>
);

/** Plain outlined tag. */
export const Chip: React.FC<{ children: React.ReactNode; tone?: "accent" | "ok" | "err" | "muted" }> = ({ children, tone = "muted" }) => {
  const col = tone === "accent" ? C.accent : tone === "ok" ? C.ok : tone === "err" ? C.err : C.muted;
  return <span style={{ fontFamily: FONT.mono, fontSize: 20, color: col, border: `1px solid ${C.line}`, borderRadius: 6, padding: "6px 12px", whiteSpace: "nowrap" }}>{children}</span>;
};

/** Right-hand feature column beside the app window. Frame f is the scene-local frame. */
export const FeatureCard: React.FC<{ from: number; to: number; n: string; eyebrow: string; title: React.ReactNode; body: string; children?: React.ReactNode }> = ({ from, to, n, eyebrow, title, body, children }) => {
  const f = useCurrentFrame();
  if (f < from - 1 || f > to + 12) return null;
  const exit = interpolate(f, [to, to + 10], [0, 1], { ...clamp, easing: EASE.in });
  return (
    <div style={{ position: "absolute", left: 1392, top: 128, width: 448, opacity: 1 - exit, transform: `translateX(${exit * -30}px)` }}>
      <Enter delay={0} y={16}><Eyebrow n={n}>{eyebrow}</Eyebrow></Enter>
      <Enter delay={5} y={26} style={{ marginTop: 28 }}><div style={{ fontFamily: FONT.sans, fontWeight: 700, fontSize: 60, lineHeight: 1.06, letterSpacing: "-0.03em", color: C.white }}>{title}</div></Enter>
      <Enter delay={11} y={20} style={{ marginTop: 26 }}><div style={{ fontFamily: FONT.sans, fontWeight: 430, fontSize: 26, lineHeight: 1.45, color: C.muted }}>{body}</div></Enter>
      <div style={{ marginTop: 34 }}>{children}</div>
    </div>
  );
};

/** Flat surface: product panel colour, one hairline border, no shadow. */
export const Panel: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 10, padding: "18px 22px", ...style }}>{children}</div>
);
