import React from "react";
import { interpolate } from "remotion";
import { C, EASE, FONT, clamp } from "../theme";
import { AppProps, AppUI, PAGE_W } from "./AppUI";

export const SCALE = 1.25;
export const WRAP_W = PAGE_W + 56;
export const VIEW_H = 600; // logical px of page visible
export const WIN = { x: 70, y: 64, w: WRAP_W * SCALE, chrome: 46 };

export interface CursorKey { f: number; x: number; y: number; click?: boolean } // logical page coords

export const Cursor: React.FC<{ frame: number; keys: CursorKey[]; scroll: number }> = ({ frame, keys, scroll }) => {
  if (keys.length === 0 || frame < keys[0].f - 8) return null;
  let x = keys[0].x, y = keys[0].y;
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i], b = keys[i + 1];
    if (frame >= a.f && frame <= b.f) {
      const t = EASE.inOut((frame - a.f) / Math.max(1, b.f - a.f));
      x = a.x + (b.x - a.x) * t; y = a.y + (b.y - a.y) * t;
    } else if (frame > b.f) { x = b.x; y = b.y; }
  }
  const vis = interpolate(frame, [keys[0].f - 8, keys[0].f], [0, 1], clamp);
  // click ripple
  const clicks = keys.filter((k) => k.click);
  const rip = clicks.map((k) => interpolate(frame, [k.f, k.f + 14], [0, 1], clamp)).filter((v) => v > 0 && v < 1);
  const pressed = clicks.some((k) => frame >= k.f && frame < k.f + 4);
  return (
    <div style={{ position: "absolute", left: x + 28, top: y - scroll, opacity: vis, pointerEvents: "none", zIndex: 20 }}>
      {rip.map((v, i) => <div key={i} style={{ position: "absolute", left: -22, top: -22, width: 44, height: 44, borderRadius: 22, border: `2px solid ${C.accent}`, opacity: 1 - v, transform: `scale(${0.3 + v})` }} />)}
      <svg width="30" height="36" viewBox="0 0 24 28" style={{ transform: `scale(${pressed ? 0.88 : 1})`, filter: "drop-shadow(0 4px 8px #000a)", transformOrigin: "2px 2px" }}>
        <path d="M2 2 L2 21 L7 16.5 L10.5 25 L13.5 23.7 L10 15.5 L17 15.5 Z" fill="#fff" stroke="#0d1117" strokeWidth="1.6" strokeLinejoin="round" />
      </svg>
    </div>
  );
};

// Browser-style window holding the replica UI.
export const AppWindow: React.FC<{ app: AppProps; scroll: number; cursor: CursorKey[]; frame: number; entrance: number; exit?: number; zoom?: { s: number; ox: number; oy: number } }> = ({ app, scroll, cursor, frame, entrance, exit = 0, zoom }) => {
  const e = entrance * (1 - exit);
  const z = zoom ?? { s: 1, ox: 50, oy: 50 };
  return (
    <div style={{ position: "absolute", left: WIN.x, top: WIN.y, width: WIN.w, opacity: e, transform: `translateY(${(1 - entrance) * 70 + exit * 30}px) scale(${(0.94 + entrance * 0.06) * z.s})`, transformOrigin: `${z.ox}% ${z.oy}%` }}>
      <div style={{ borderRadius: 16, overflow: "hidden", border: `1px solid ${C.line}`, background: C.bg, boxShadow: "none" }}>
        <div style={{ height: WIN.chrome, background: C.panel, borderBottom: `1px solid ${C.line}`, display: "flex", alignItems: "center", gap: 10, padding: "0 18px" }}>
          {["#f85149", "#d29922", "#3fb950"].map((c, i) => <div key={i} style={{ width: 13, height: 13, borderRadius: 7, background: c, opacity: 0.85 }} />)}
          <div style={{ marginLeft: 18, flex: 1, height: 28, borderRadius: 8, background: C.bg, border: `1px solid ${C.line}`, display: "flex", alignItems: "center", padding: "0 14px", fontFamily: FONT.mono, fontSize: 14, color: C.muted, gap: 8 }}>
            <span style={{ color: C.ok }}>●</span> tinyattention <span style={{ opacity: 0.5 }}>/</span><span style={{ color: C.fg }}>app</span>
          </div>
        </div>
        <div style={{ position: "relative", width: WIN.w, height: VIEW_H * SCALE, overflow: "hidden", background: C.bg }}>
          <div style={{ position: "absolute", left: 0, top: 0, width: WRAP_W, transform: `scale(${SCALE}) translateY(${-scroll}px)`, transformOrigin: "0 0" }}>
            <div style={{ padding: "28px 28px 60px" }}><AppUI {...app} /></div>
          </div>
          <div style={{ position: "absolute", left: 0, top: 0, width: WRAP_W, height: VIEW_H, transform: `scale(${SCALE})`, transformOrigin: "0 0", pointerEvents: "none" }}>
            <Cursor frame={frame} keys={cursor} scroll={scroll - 28} />
          </div>
        </div>
      </div>
    </div>
  );
};
