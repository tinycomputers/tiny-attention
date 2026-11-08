// One continuous app window across the four product scenes (tiles, engine, memory, trust). Frame f is relative to the
// start of the "tiles" scene. Every number shown is from the recorded engine trace.
import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { C, EASE, FONT, clamp } from "../theme";
import { AppWindow, CursorKey } from "../app/AppWindow";
import { GEO, Tab } from "../app/AppUI";
import { CMP, RUN, SWEEP, TOTAL_CLOCKS, snapAt } from "../app/sim";
import { SCENES } from "../timeline";
import { Chip, Enter, FeatureCard, Panel } from "../components/bits";

const B = SCENES.tiles.from;
export const E = SCENES.engine.from - B;
export const M = SCENES.memory.from - B;
export const R = SCENES.trust.from - B;
export const END = SCENES.cta.from - B;

const kf = (f: number, keys: [number, number][]) => {
  if (f <= keys[0][0]) return keys[0][1];
  for (let i = 0; i < keys.length - 1; i++) {
    const [a, va] = keys[i], [b, vb] = keys[i + 1];
    if (f <= b) return va + (vb - va) * EASE.inOut((f - a) / (b - a));
  }
  return keys[keys.length - 1][1];
};

const BTN = { y: 243, start: 32, back: 108, step: 188, run: 284, reset: 376 };
const cmpBtn = { run: { x: 92, y: 599 }, sweep: { x: 253, y: 599 } };
// cursor: logical AppUI coords, stage frames
const CURSOR: CursorKey[] = [
  { f: 8, x: 700, y: 420 },
  { f: 40, x: BTN.run, y: BTN.y },
  { f: 44, x: BTN.run, y: BTN.y, click: true },
  { f: 58, x: BTN.run + 6, y: BTN.y + 4 },
  { f: 150, x: 120, y: 690 },
  { f: 176, x: 86, y: 652 },
  { f: 250, x: 86, y: 652 },
  { f: E + 2, x: GEO.tabs.tile, y: GEO.tabs.y },
  { f: E + 6, x: GEO.tabs.tile, y: GEO.tabs.y, click: true },
  { f: E + 40, x: 640, y: 700 },
  { f: M - 4, x: 640, y: 700 },
  { f: M + 4, x: GEO.tabs.compare, y: GEO.tabs.y },
  { f: M + 8, x: GEO.tabs.compare, y: GEO.tabs.y, click: true },
  { f: M + 46, x: cmpBtn.run.x, y: cmpBtn.run.y },
  { f: M + 50, x: cmpBtn.run.x, y: cmpBtn.run.y, click: true },
  { f: M + 126, x: cmpBtn.sweep.x, y: cmpBtn.sweep.y },
  { f: M + 130, x: cmpBtn.sweep.x, y: cmpBtn.sweep.y, click: true },
  { f: M + 160, x: 640, y: 900 },
  { f: R - 2, x: 640, y: 900 },
  { f: R + 14, x: GEO.tabs.schedule, y: GEO.tabs.y },
  { f: R + 18, x: GEO.tabs.schedule, y: GEO.tabs.y, click: true },
  { f: R + 46, x: BTN.reset, y: BTN.y },
  { f: R + 50, x: BTN.reset, y: BTN.y, click: true },
  { f: R + 60, x: BTN.step, y: BTN.y },
  { f: R + 64, x: BTN.step, y: BTN.y, click: true },
  { f: R + 74, x: BTN.step, y: BTN.y, click: true },
  { f: R + 84, x: BTN.step, y: BTN.y, click: true },
  { f: R + 94, x: BTN.step, y: BTN.y, click: true },
  { f: R + 110, x: BTN.run, y: BTN.y },
  { f: R + 114, x: BTN.run, y: BTN.y, click: true },
  { f: R + 140, x: 560, y: 330 },
];
export const CLICK_FRAMES = CURSOR.filter((c) => c.click).map((c) => c.f);

export const AppStage: React.FC = () => {
  const f = useCurrentFrame();
  const entrance = interpolate(f, [0, 26], [0, 1], { ...clamp, easing: EASE.out });
  const exit = interpolate(f, [END - 10, END], [0, 1], { ...clamp, easing: EASE.in });

  // ---- tab & scroll
  let tab: Tab = "schedule";
  if (f >= E + 8) tab = "tile";
  if (f >= M + 10) tab = "compare";
  if (f >= R + 20) tab = "schedule";
  const scroll = kf(f, [[0, 0], [60, 0], [135, 292], [E, 292], [E + 40, 540], [E + 96, 540], [E + 130, 770], [M - 34, 770], [M - 6, 540], [M, 540], [R, 540], [R + 16, 300], [R + 44, 0], [END, 0]]);

  // ---- clock
  let clock = 0;
  const run1 = interpolate(f, [48, 255], [0, TOTAL_CLOCKS], { ...clamp, easing: EASE.inOut });
  const tt: any[] = RUN.tileTrace;
  const k = Math.min(5, Math.max(0, Math.floor((f - (E + 40)) / 34)));
  if (f < E + 40) clock = run1;
  else if (f < R) clock = tt[k].clk + 1;
  else {
    if (f < R + 50) clock = TOTAL_CLOCKS;
    else if (f < R + 64) clock = 0;
    else if (f < R + 114) clock = Math.min(4, 1 + Math.floor((f - (R + 64)) / 10));
    else clock = Math.min(TOTAL_CLOCKS, interpolate(f, [R + 116, R + 170], [4, TOTAL_CLOCKS], { ...clamp, easing: EASE.out }));
  }
  const s = snapAt(clock);

  // ---- engine glow
  const u = (f - (E + 40)) % 34;
  const glow = [0, 1, 2, 3, 4].map((i) => (f < E + 40 ? 0 : interpolate(u, [i * 5, i * 5 + 4, i * 5 + 11], [0, 1, 0], clamp)));

  // ---- compare
  const cmpP = interpolate(f, [M + 54, M + 100], [0, 1], { ...clamp, easing: EASE.out });
  const sweepP = interpolate(f, [M + 134, M + 196], [0, 1], { ...clamp, easing: EASE.out });
  const btnAt = (a: number) => f >= a && f < a + 14;
  const btn = btnAt(44) ? "Run to end" : btnAt(R + 50) ? "Reset" : [R + 64, R + 74, R + 84, R + 94].some(btnAt) ? "Step" : btnAt(R + 114) ? "Run to end" : undefined;

  const pulse = 0.5 + 0.5 * Math.sin(f / 5);
  return (
    <AbsoluteFill>
      <AppWindow frame={f} entrance={entrance} exit={exit} scroll={scroll} cursor={CURSOR}
        app={{ s, tab, btn, glow, pulse, cmpP, cmpShown: f >= M + 50, sweepP, cmpBtn: btnAt(M + 50) ? 1 : btnAt(M + 130) ? 2 : 0 }} />

      {/* callout: masked tiles */}
      <MaskedCallout f={f} scroll={scroll} />

      <div style={{ opacity: 1 - exit }}>
        <FeatureCard from={16} to={E - 4} n="01" eyebrow="TILED SCHEDULE" title={<>One tile at a <span style={{ color: C.accent }}>time.</span></>} body="Four by four, in order. Masked tiles are never computed, and the full matrix is never stored.">
          <SoftmaxMini s={s} f={f} />
        </FeatureCard>
        <ShiftedCard from={E + 4} to={M - 4} f={f}>
          <FeatureCard from={0} to={M - E - 8} n="02" eyebrow="THE DATAPATH" title={<>Real hardware, <span style={{ color: C.accent }}>live.</span></>} body="Two INT8 systolic arrays, an online softmax unit and a rescaling accumulator. Every value is a debug port on the RTL.">
            <Stages glow={glow} />
          </FeatureCard>
        </ShiftedCard>
        <ShiftedCard from={M + 4} to={R - 4} f={f}>
          <FeatureCard from={0} to={R - M - 8} n="03" eyebrow="NAIVE VS FLASH" title={<>Nearly 4× less <span style={{ color: C.accent }}>memory.</span></>} body="Same inputs, both schedules. FLASH never writes the N×N matrix out to HBM.">
            <Ratio f={f - M} cmpP={cmpP} sweepP={sweepP} />
          </FeatureCard>
        </ShiftedCard>
        <ShiftedCard from={R + 6} to={END} f={f}>
          <FeatureCard from={0} to={END - R - 6} n="04" eyebrow="VERIFIED" title={<>Bit-exact. <span style={{ color: C.accent }}>Every run.</span></>} body="Checked against a golden model, and you can step it clock by clock.">
            <Checks f={f - (R + 6)} />
          </FeatureCard>
        </ShiftedCard>
      </div>
    </AbsoluteFill>
  );
};

/** Re-bases the frame for a FeatureCard so its entrance starts at `from`. */
const ShiftedCard: React.FC<{ from: number; to: number; f: number; children: React.ReactNode }> = ({ from, to, f, children }) => {
  if (f < from || f > to + 12) return null;
  return <FrameShift offset={from}>{children}</FrameShift>;
};
import { Sequence } from "remotion";
const FrameShift: React.FC<{ offset: number; children: React.ReactNode }> = ({ offset, children }) => (
  <Sequence from={offset} layout="none">{children}</Sequence>
);

const MaskedCallout: React.FC<{ f: number; scroll: number }> = ({ f, scroll }) => {
  const a = interpolate(f, [150, 168, 236, 250], [0, 1, 1, 0], { ...clamp, easing: EASE.out });
  if (a <= 0) return null;
  // logical (AppUI) coords of the masked cells (q0, kv1..2) -> screen
  const sx = (lx: number) => 70 + (lx + 28) * 1.25;
  const sy = (ly: number) => 64 + 46 + (ly + 28 - scroll) * 1.25;
  return (
    <div style={{ position: "absolute", left: sx(176), top: sy(650) - 30, opacity: a, transform: `translateX(${(1 - a) * -16}px)` }}>
      <svg width="120" height="60" style={{ position: "absolute", left: -112, top: 6 }}>
        <path d="M110 30 C 70 30, 50 30, 6 30" stroke={C.accent} strokeWidth="2.5" fill="none" strokeDasharray="6 5" />
        <path d="M14 22 L4 30 L14 38" stroke={C.accent} strokeWidth="2.5" fill="none" />
      </svg>
      <div style={{ fontFamily: FONT.sans, fontSize: 26, fontWeight: 650, color: C.white, background: `${C.panel}f2`, border: `1px solid ${C.accent}`, borderRadius: 12, padding: "10px 18px", boxShadow: "none", whiteSpace: "nowrap" }}>
        Masked: never computed
      </div>
    </div>
  );
};

const SoftmaxMini: React.FC<{ s: ReturnType<typeof snapAt>; f: number }> = ({ s, f }) => {
  const t = s.last;
  const rows = [0, 1, 2, 3];
  return (
    <Enter delay={22} y={30}>
      <Panel style={{ padding: "16px 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontFamily: FONT.mono, fontSize: 18, color: C.muted, marginBottom: 10 }}>
          <span>online softmax · running state</span><span style={{ color: C.accent }}>q{s.done > 0 ? ORDERQ(s.done - 1) : 0}</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "0.7fr 1fr 1fr 1fr", gap: "6px 10px", fontFamily: FONT.mono, fontSize: 21, fontVariantNumeric: "tabular-nums" }}>
          <span style={{ color: C.muted }}>row</span><span style={{ color: C.muted }}>m</span><span style={{ color: C.muted }}>α</span><span style={{ color: C.muted }}>l</span>
          {rows.map((r) => (
            <React.Fragment key={r}>
              <span style={{ color: C.muted }}>{r}</span>
              <span style={{ color: C.white }}>{t ? (t.m[r] / 256).toFixed(2) : "0.00"}</span>
              <span style={{ color: C.fg }}>{t ? (t.alpha[r] / 32768).toFixed(3) : "0.000"}</span>
              <span style={{ color: C.white }}>{t ? (t.l[r] / 4096).toFixed(2) : "0.00"}</span>
            </React.Fragment>
          ))}
        </div>
      </Panel>
    </Enter>
  );
};
const ORDERQ = (i: number) => [0, 1, 1, 2, 2, 2][i] ?? 0;

const Stages: React.FC<{ glow: number[] }> = ({ glow }) => {
  const names = ["MM1 · S = Q·Kᵀ", "Scale and mask", "Online softmax", "MM2 · P·V", "O accumulator"];
  return (
    <Panel style={{ padding: "14px 20px" }}>
      {names.map((n, i) => (
        <Enter key={i} delay={18 + i * 5} y={14} x={20}>
          <div style={{ display: "flex", alignItems: "center", gap: 16, padding: "8px 0", fontFamily: FONT.mono, fontSize: 22, color: glow[i] > 0.3 ? C.white : C.muted, borderBottom: i < 4 ? `1px solid ${C.line}66` : "none" }}>
            <span style={{ width: 14, height: 14, borderRadius: 7, background: glow[i] > 0.3 ? C.accent : C.line, boxShadow: "none" }} />
            {n}
          </div>
        </Enter>
      ))}
    </Panel>
  );
};

const Ratio: React.FC<{ f: number; cmpP: number; sweepP: number }> = ({ f, cmpP, sweepP }) => {
  const a = CMP.naive.rd + CMP.naive.wr, b = CMP.flash.rd + CMP.flash.wr;
  const last = SWEEP[SWEEP.length - 1];
  const r1 = a / b, r2 = last.naiveBytes / last.flashBytes;
  const ratio = r1 + (r2 - r1) * EASE.inOut(sweepP);
  const lab = sweepP > 0.02 ? "at N = 32" : "this run";
  return (
    <Enter delay={22} y={30}>
      <Panel style={{ padding: "18px 24px" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
          <span style={{ fontFamily: FONT.sans, fontWeight: 800, fontSize: 104, letterSpacing: "-0.04em", color: C.accent, textShadow: "none", fontVariantNumeric: "tabular-nums", opacity: cmpP }}>{(ratio * Math.min(1, cmpP * 1.01)).toFixed(1)}×</span>
          <span style={{ fontFamily: FONT.mono, fontSize: 22, color: C.muted }}>{lab}</span>
        </div>
        <div style={{ fontFamily: FONT.mono, fontSize: 21, color: C.muted, marginTop: 4, fontVariantNumeric: "tabular-nums" }}>
          <span style={{ color: C.err }}>{Math.round(a * cmpP).toLocaleString("en-US")} B</span> naive → <span style={{ color: C.accent }}>{Math.round(b * cmpP).toLocaleString("en-US")} B</span> flash
        </div>
      </Panel>
    </Enter>
  );
};

const Checks: React.FC<{ f: number }> = ({ f }) => {
  const items = ["Output O, every value", "HBM bytes read and written", "Tiles and K/V loads", "Busy cycles"];
  return (
    <Panel style={{ padding: "14px 22px" }}>
      {items.map((t, i) => {
        const a = interpolate(f, [26 + i * 12, 38 + i * 12], [0, 1], { ...clamp, easing: EASE.back });
        return (
          <Enter key={i} delay={16 + i * 6} y={12}>
            <div style={{ display: "flex", alignItems: "center", gap: 16, padding: "9px 0", fontFamily: FONT.sans, fontSize: 25, color: C.fg, borderBottom: i < 3 ? `1px solid ${C.line}66` : "none" }}>
              <span style={{ width: 30, height: 30, borderRadius: 15, background: `${C.ok}${a > 0.5 ? "" : "33"}`, border: `2px solid ${C.ok}`, display: "flex", alignItems: "center", justifyContent: "center", color: C.bg, transform: `scale(${0.6 + a * 0.4})`, boxShadow: "none" }}>
                <svg width="16" height="16" viewBox="0 0 16 16"><path d="M3 8.5 L6.5 12 L13 4.5" stroke={a > 0.5 ? C.bg : "transparent"} strokeWidth="2.6" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </span>
              {t}
            </div>
          </Enter>
        );
      })}
      <div style={{ marginTop: 12, display: "flex", gap: 10, flexWrap: "wrap" }}><Chip tone="ok">equal to golden model</Chip></div>
    </Panel>
  );
};
