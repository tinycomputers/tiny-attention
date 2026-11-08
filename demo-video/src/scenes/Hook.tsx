import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { C, EASE, FONT, clamp, sp } from "../theme";
import { Enter, Eyebrow } from "../components/bits";
import { SWEEP } from "../app/sim";

const rnd = (i: number, j: number) => { const x = Math.sin(i * 127.1 + j * 311.7) * 43758.5453; return x - Math.floor(x); };
const SIZES = [{ n: 4, at: 4 }, { n: 8, at: 30 }, { n: 16, at: 56 }, { n: 32, at: 82 }];
const bytesAt = (n: number) => SWEEP.find((r) => r.n === n)!.naiveBytes;

export const Hook: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const ex = interpolate(f, [dur - 10, dur], [0, 1], { ...clamp, easing: EASE.in });
  const cur = [...SIZES].reverse().find((s) => f >= s.at) ?? SIZES[0];
  const n = cur.n;
  const MX = 520, cell = MX / 32;
  const cells: React.ReactNode[] = [];
  for (let i = 0; i < 32; i++) for (let j = 0; j < 32; j++) {
    const shown = i < n && j < n;
    const born = SIZES.find((s) => s.n === n)!.at;
    const d = (i + j) * 0.5;
    const a = shown ? interpolate(f, [born + d * 0.35, born + d * 0.35 + 7], [0, 1], { ...clamp, easing: EASE.out }) : 0;
    const heat = Math.min(1, (Math.exp(-Math.abs(i - j) / 5) * 0.8 + rnd(i, j) * 0.45));
    cells.push(<rect key={i * 32 + j} x={j * cell + 1} y={i * cell + 1} width={cell - 2} height={cell - 2} rx={2} fill={C.accent} opacity={a * (0.12 + heat * 0.88)} />);
  }
  const bytes = Math.round(interpolate(f, [cur.at, cur.at + 18], [n === 4 ? 0 : bytesAt(n / 2), bytesAt(n)], { ...clamp, easing: EASE.out }));
  const pct = interpolate(bytes, [0, bytesAt(32)], [0, 1], clamp);
  const scaleOut = 1 - ex * 0.03;
  const mp = sp(f, fps, 6, "soft");
  const prevN = cur.n === 4 ? 4 : cur.n / 2;
  const boxW = (interpolate(f, [cur.at, cur.at + 16], [prevN, n], { ...clamp, easing: EASE.out }) * cell) + 40;
  return (
    <AbsoluteFill style={{ opacity: 1 - ex, transform: `scale(${scaleOut})` }}>
      {/* headline */}
      <div style={{ position: "absolute", left: 130, top: 238 }}>
        <Enter delay={2} y={16}><Eyebrow>FLASHATTENTION · IN HARDWARE</Eyebrow></Enter>
        <div style={{ marginTop: 34, fontFamily: FONT.sans, fontWeight: 800, fontSize: 112, lineHeight: 1.04, letterSpacing: "-0.045em", color: C.white }}>
          {["Attention", "has", "a"].map((w, i) => <Enter key={i} delay={6 + i * 5} y={50} style={{ display: "inline-block", marginRight: 26 }}>{w}</Enter>)}
          <br />
          {["memory", "problem."].map((w, i) => (
            <Enter key={i} delay={24 + i * 7} y={60} cfg="pop" style={{ display: "inline-block", marginRight: 26, color: C.accent, textShadow: "none" }}>{w}</Enter>
          ))}
        </div>
        <Enter delay={52} y={22} style={{ marginTop: 40 }}>
          <div style={{ fontFamily: FONT.sans, fontSize: 34, color: C.muted, fontWeight: 450, maxWidth: 800, lineHeight: 1.4 }}>
            Scores grow as <span style={{ color: C.fg, fontFamily: FONT.mono }}>N × N</span>. Store them all, and memory traffic explodes.
          </div>
        </Enter>
      </div>
      {/* matrix */}
      <div style={{ position: "absolute", left: 1230, top: 190, width: 560, opacity: mp, transform: `translateY(${(1 - mp) * 40}px)` }}>
        <div style={{ fontFamily: FONT.mono, fontSize: 22, color: C.muted, display: "flex", justifyContent: "space-between", marginBottom: 14 }}>
          <span>scores S = Q·Kᵀ</span><span style={{ color: C.white, fontWeight: 600 }}>N = {n}</span>
        </div>
        <div style={{ height: 560 }}>
        <div style={{ width: boxW, height: boxW, background: C.panel, border: `1px solid ${C.line}`, borderRadius: 14, padding: 20, overflow: "hidden", boxSizing: "border-box", boxShadow: "none" }}>
          <svg width={MX} height={MX} style={{ display: "block" }}>{cells}</svg>
        </div>
        </div>
        <div style={{ marginTop: 22 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontFamily: FONT.mono, fontSize: 22 }}>
            <span style={{ color: C.muted }}>NAIVE · HBM traffic</span>
            <span style={{ color: C.err, fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>{bytes.toLocaleString("en-US")} B</span>
          </div>
          <div style={{ marginTop: 10, height: 12, borderRadius: 6, background: C.line, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${pct * 100}%`, background: C.err, boxShadow: "none" }} />
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
