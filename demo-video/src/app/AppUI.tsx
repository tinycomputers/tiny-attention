// Faithful replica of web/src/components/Visualizer.tsx (same structure, tokens, copy). Fixed block heights keep layout
// deterministic so the cursor can target buttons. Logical width 940 px; the window scales it up.
import React from "react";
import { C, FONT } from "../theme";
import { CMP, EXPECTED, LABELS, ORDER, SWEEP, Snap } from "./sim";

export type Tab = "softmax" | "tile" | "schedule" | "compare";
const muted: React.CSSProperties = { color: C.muted };
const panel: React.CSSProperties = { background: C.panel };
export const PAGE_W = 940;

// ---- fixed geometry (logical px), used by the cursor
export const GEO = {
  buttons: { y: 245, run: { x: 284 }, start: { x: 32 } },
  tabs: { y: 544, softmax: 62, tile: 192, schedule: 310, compare: 439 },
  tabPanelTop: 586,
  schedulePreset: { x: 95, y: 112 },
};

const Btn: React.FC<{ w: number; children: React.ReactNode; active?: boolean; disabled?: boolean }> = ({ w, children, active, disabled }) => (
  <div style={{ ...panel, width: w, height: 30, borderRadius: 4, fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center", opacity: disabled ? 0.45 : 1, outline: active ? `2px solid ${C.accent}` : "none", boxShadow: "none" }}>{children}</div>
);
const Select: React.FC<{ w: number; v: string }> = ({ w, v }) => (
  <div style={{ ...panel, width: w, height: 28, borderRadius: 4, padding: "0 8px", display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 14, border: `1px solid ${C.line}` }}>
    <span>{v}</span><span style={{ ...muted, fontSize: 10 }}>⌄</span>
  </div>
);
const Check: React.FC<{ on?: boolean; label: string }> = ({ on, label }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 6, height: 28, alignSelf: "flex-end" }}>
    <div style={{ width: 14, height: 14, borderRadius: 3, background: on ? C.accent : "transparent", border: `1px solid ${on ? C.accent : C.muted}`, display: "flex", alignItems: "center", justifyContent: "center", color: C.bg, fontSize: 10, fontWeight: 800 }}>{on ? "✓" : ""}</div>
    <span>{label}</span>
  </div>
);
const Card: React.FC<{ label: string; children: React.ReactNode; mono?: boolean }> = ({ label, children, mono }) => (
  <div style={{ ...panel, borderRadius: 4, padding: 12, height: 64, boxSizing: "border-box" }}>
    <div style={{ ...muted, fontSize: 14, lineHeight: "20px" }}>{label}</div>
    <div style={{ fontSize: 18, lineHeight: "26px", fontVariantNumeric: "tabular-nums", fontFamily: mono ? FONT.mono : FONT.sans }}>{children}</div>
  </div>
);

const HbmBar: React.FC<{ label: string; value: number; total: number; color: string }> = ({ label, value, total, color }) => (
  <div style={{ height: 34 }}>
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontVariantNumeric: "tabular-nums", lineHeight: "16px" }}>
      <span style={muted}>{label}</span><span>{Math.round(value)} / {total} B</span>
    </div>
    <div style={{ marginTop: 4, height: 8, width: "100%", borderRadius: 4, overflow: "hidden", background: C.panel }}>
      <div style={{ height: "100%", width: `${total > 0 ? Math.min(100, (value / total) * 100) : 0}%`, background: color }} />
    </div>
  </div>
);

const TAB_DEFS: { id: Tab; label: string; w: number }[] = [
  { id: "softmax", label: "L1 Softmax cell", w: 124 }, { id: "tile", label: "L2 Tile dataflow", w: 128 },
  { id: "schedule", label: "L3 Schedule", w: 100 }, { id: "compare", label: "L4 Naive vs Flash", w: 150 },
];
const Tabs: React.FC<{ tab: Tab }> = ({ tab }) => (
  <div style={{ display: "flex", gap: 4, borderBottom: `1px solid ${C.line}`, height: 36 }}>
    {TAB_DEFS.map((t) => {
      const sel = t.id === tab;
      return (
        <div key={t.id} style={{ width: t.w, height: 36, boxSizing: "border-box", borderRadius: "6px 6px 0 0", border: `1px solid ${sel ? C.accent : "transparent"}`, borderBottom: "none", color: sel ? C.fg : C.muted, background: sel ? C.panel : "transparent", fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center" }}>{t.label}</div>
      );
    })}
  </div>
);

// ---------- L3
const cellOf = (qi: number, kj: number, s: Snap) => {
  const idx = ORDER.findIndex((o) => o.qi === qi && o.kj === kj);
  if (kj > qi) return { bg: "transparent", dot: false, op: 0.25 };
  if (idx >= 0 && idx < s.done) return { bg: C.ok, dot: false, op: 1 };
  if (idx === s.inFlight) return { bg: C.accent, dot: true, op: 1 };
  return { bg: C.panel, dot: false, op: 1 };
};
const Schedule: React.FC<{ s: Snap; pulse?: number }> = ({ s, pulse = 0 }) => {
  const nb = 3;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 24 }}>
      <section>
        <h2 style={{ margin: "0 0 8px", fontSize: 14, fontWeight: 600, lineHeight: "20px" }}>Tile schedule (row groups × K/V tiles)</h2>
        <div style={{ display: "inline-grid", gap: 4, gridTemplateColumns: `auto repeat(${nb}, 36px)` }}>
          <div />
          {[0, 1, 2].map((k) => <div key={k} style={{ ...muted, textAlign: "center", fontSize: 12 }}>kv{k}</div>)}
          {[0, 1, 2].map((qi) => (
            <React.Fragment key={qi}>
              <div style={{ ...muted, textAlign: "right", paddingRight: 8, fontSize: 12, alignSelf: "center" }}>q{qi}</div>
              {[0, 1, 2].map((kj) => {
                const c = cellOf(qi, kj, s);
                const flight = c.dot;
                return (
                  <div key={kj} style={{ height: 36, borderRadius: 4, background: c.bg, border: `1px solid ${C.muted}`, opacity: c.op, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: C.bg, boxShadow: "none" }}>
                    {c.dot ? "●" : ""}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
        <p style={{ ...muted, margin: "8px 0 0", fontSize: 12, lineHeight: "16px" }}>Green: finished. Blue: in flight. Faded: masked by the causal mask.</p>
      </section>
    </div>
  );
};

const KV: React.FC<{ s: Snap }> = ({ s }) => {
  const kv = s.last?.kv;
  const tile = (title: string, vals: number[]) => (
    <div>
      <div style={{ ...muted, fontSize: 10 }}>{title}</div>
      <div style={{ display: "inline-grid", gap: "0 8px", gridTemplateColumns: "repeat(4, auto)", fontFamily: FONT.mono, fontSize: 12, fontVariantNumeric: "tabular-nums" }}>
        {vals.map((v, i) => <span key={i}>{v}</span>)}
      </div>
    </div>
  );
  return (
    <section>
      <h2 style={{ margin: "0 0 8px", fontSize: 14, fontWeight: 600, lineHeight: "20px" }}>On-chip K/V buffers</h2>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {[0, 1].map((b) => (
          <div key={b} style={{ ...panel, borderRadius: 4, padding: 8, display: "flex", flexDirection: "column", gap: 8, outline: kv && kv.sel === b ? `2px solid ${C.accent}` : "none" }}>
            <div style={{ ...muted, fontSize: 12 }}>Buffer {b}{kv && kv.sel === b ? " · in use" : ""}</div>
            {tile("K tile", kv ? kv.k.slice(b * 16, b * 16 + 16) : new Array(16).fill(0))}
            {tile("V tile", kv ? kv.v.slice(b * 16, b * 16 + 16) : new Array(16).fill(0))}
          </div>
        ))}
      </div>
    </section>
  );
};

// ---------- L2
const Grid: React.FC<{ title: string; values: number[]; scale?: number; digits?: number }> = ({ title, values, scale = 1, digits = 0 }) => (
  <div>
    <div style={{ ...muted, fontSize: 10 }}>{title}</div>
    <div style={{ display: "inline-grid", gap: "0 8px", gridTemplateColumns: "repeat(4, auto)", fontFamily: FONT.mono, fontSize: 11, fontVariantNumeric: "tabular-nums" }}>
      {values.slice(0, 16).map((v, i) => <span key={i}>{(v * scale).toFixed(digits)}</span>)}
    </div>
  </div>
);
const Stage: React.FC<{ name: string; glow: number; children: React.ReactNode }> = ({ name, glow, children }) => (
  <div style={{ ...panel, borderRadius: 4, padding: 12, border: `1px solid ${glow > 0.05 ? C.accent : C.line}`, boxShadow: "none", display: "flex", flexDirection: "column", gap: 12 }}>
    <div style={{ fontSize: 12, fontWeight: 600, marginBottom: -4 }}>{name}</div>
    {children}
  </div>
);
const TileDataflow: React.FC<{ s: Snap; glow: number[] }> = ({ s, glow }) => {
  const t = s.last;
  const dp = t?.dp ?? {};
  const z = (a: any) => a ?? new Array(16).fill(0);
  return (
    <div>
      <p style={{ ...muted, margin: "0 0 12px", fontSize: 12, lineHeight: "16px" }}>Wires at the last busy clock of the run so far (between tiles they are idle). The S, Z, P and O grids are for the row group of the most recent tile.</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
        <Stage name="MM1: S = Q · Kᵀ (array)" glow={glow[0]}>
          <Grid title="Kᵀ weights (int8)" values={z(dp.w1)} />
          <Grid title="Q activations (int8)" values={z(dp.a1)} />
          <Grid title="S partial sums (int17)" values={z(dp.p1)} />
        </Stage>
        <Stage name="Scale and mask" glow={glow[1]}>
          <Grid title="Z = S · scale (Q8.8, log2)" values={z(t?.z)} scale={1 / 256} digits={2} />
        </Stage>
        <Stage name="Online softmax (row max, exp2, sum)" glow={glow[2]}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
            <Grid title="m" values={t?.m ?? [0, 0, 0, 0]} scale={1 / 256} digits={2} />
            <Grid title="α" values={t?.alpha ?? [0, 0, 0, 0]} scale={1 / 32768} digits={3} />
            <Grid title="l" values={t?.l ?? [0, 0, 0, 0]} scale={1 / 4096} digits={2} />
          </div>
          <Grid title="P = 2^(Z − m) (UQ1.8)" values={z(t?.p)} scale={1 / 256} digits={3} />
        </Stage>
        <Stage name="MM2: P · V (array)" glow={glow[3]}>
          <Grid title="V weights (int8)" values={z(dp.w2)} />
          <Grid title="P activations (uint9)" values={z(dp.a2)} scale={1 / 256} digits={3} />
          <Grid title="P·V partial sums (int18)" values={z(dp.p2)} />
        </Stage>
        <Stage name="O accumulator (rescaled by α)" glow={glow[4]}>
          <Grid title="O (Q8.8 accumulator, V-LSB)" values={z(t?.o)} scale={1 / 256} digits={1} />
        </Stage>
      </div>
    </div>
  );
};

// ---------- L4
const CBar: React.FC<{ label: string; a: number; b: number; p: number }> = ({ label, a, b, p }) => {
  const max = Math.max(a, b, 1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ ...muted, fontSize: 12 }}>{label}</div>
      {[{ n: "FLASH", v: a, c: C.accent }, { n: "NAIVE", v: b, c: C.err }].map((r) => (
        <div key={r.n} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, fontVariantNumeric: "tabular-nums" }}>
          <span style={{ ...muted, width: 48 }}>{r.n}</span>
          <div style={{ flex: 1, height: 12, borderRadius: 4, background: C.line }}>
            <div style={{ height: 12, borderRadius: 4, width: `${(r.v / max) * 100 * p}%`, background: r.c }} />
          </div>
          <span style={{ width: 70, textAlign: "right" }}>{Math.round(r.v * p)}{label.startsWith("HBM") ? " B" : ""}</span>
        </div>
      ))}
    </div>
  );
};
const SweepChart: React.FC<{ p: number }> = ({ p }) => {
  const Wd = 360, Ht = 150, maxB = Math.max(...SWEEP.flatMap((r) => [r.flashBytes, r.naiveBytes]));
  const x = (n: number) => 10 + ((n - 4) / 28) * (Wd - 20);
  const y = (b: number) => Ht - 14 - (b / maxB) * (Ht - 28);
  const n = Math.max(2, Math.ceil(p * SWEEP.length));
  const pts = (k: "flashBytes" | "naiveBytes") => SWEEP.slice(0, n).map((r) => `${x(r.n)},${y(r[k])}`).join(" ");
  return (
    <div>
      <svg viewBox={`0 0 ${Wd} ${Ht}`} style={{ width: 440, display: "block" }}>
        <polyline points={pts("naiveBytes")} fill="none" stroke={C.err} strokeWidth={2} />
        <polyline points={pts("flashBytes")} fill="none" stroke={C.accent} strokeWidth={2} />
        {SWEEP.map((r) => <text key={r.n} x={x(r.n)} y={Ht - 2} fontSize={9} textAnchor="middle" fill={C.muted}>{r.n}</text>)}
      </svg>
      <div style={{ display: "flex", gap: 16, fontSize: 12, ...muted }}>
        <span><span style={{ color: C.accent }}>━</span> FLASH HBM bytes</span><span><span style={{ color: C.err }}>━</span> NAIVE HBM bytes</span><span>x-axis: N</span>
      </div>
    </div>
  );
};
const Compare: React.FC<{ p: number; sweepP: number; shown: boolean; btn: number }> = ({ p, sweepP, shown, btn }) => {
  const f = CMP.flash, n = CMP.naive;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div style={{ display: "flex", gap: 8, fontSize: 14, alignItems: "center" }}>
        <div style={{ ...panel, borderRadius: 4, padding: "0 12px", height: 30, display: "flex", alignItems: "center", outline: btn === 1 ? `2px solid ${C.accent}` : "none", boxShadow: "none" }}>Run both on these inputs</div>
        <div style={{ ...panel, borderRadius: 4, padding: "0 12px", height: 30, display: "flex", alignItems: "center", outline: btn === 2 ? `2px solid ${C.accent}` : "none", boxShadow: "none" }}>Sweep N = 4 to 32</div>
        <span style={{ ...muted, fontSize: 12 }}>FLASH: Q block 4 rows. NAIVE: 4-row blocks. Causal on.</span>
      </div>
      {shown && (
        <section style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>Same inputs, both schedules</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <CBar label="HBM bytes (read + write)" a={f.rd + f.wr} b={n.rd + n.wr} p={p} />
            <CBar label="Busy cycles" a={f.cycles} b={n.cycles} p={p} />
            <CBar label="Tiles computed" a={f.tiles} b={n.tiles} p={p} />
            <CBar label="K/V tile loads" a={f.kvLoads} b={n.kvLoads} p={p} />
          </div>
        </section>
      )}
      {sweepP > 0 && <section style={{ display: "flex", flexDirection: "column", gap: 8 }}><h3 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>HBM bytes against N</h3><SweepChart p={sweepP} /></section>}
    </div>
  );
};

export interface AppProps {
  s: Snap; tab: Tab; running?: boolean; btn?: string; glow?: number[]; pulse?: number;
  cmpP?: number; cmpShown?: boolean; sweepP?: number; cmpBtn?: number;
}
export const AppUI: React.FC<AppProps> = ({ s, tab, btn, glow = [0, 0, 0, 0, 0], pulse = 0, cmpP = 0, cmpShown = false, sweepP = 0, cmpBtn = 0 }) => {
  const finished = s.state === "DONE";
  return (
    <div style={{ width: PAGE_W, padding: "0 0", color: C.fg, fontFamily: FONT.sans, fontSize: 16, lineHeight: 1.5 }}>
      <div style={{ height: 64 }}>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, lineHeight: "32px", color: C.fg }}>TinyAttention visualizer</h1>
        <p style={{ ...muted, margin: "4px 0 0", fontSize: 14, lineHeight: "20px" }}>Every number here is read from the compiled RTL, running in your browser. Pick a preset, press Start, then step through the clock.</p>
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 24, height: 48, alignItems: "flex-end", fontSize: 14 }}>
        <div><div style={{ ...muted, lineHeight: "20px" }}>Preset</div><Select w={190} v="toy_sentence (N=12)" /></div>
        <div><div style={{ ...muted, lineHeight: "20px" }}>Schedule</div><Select w={88} v="FLASH" /></div>
        <Check on label="Causal" />
        <Check label="Overlap" />
        <div><div style={{ ...muted, lineHeight: "20px" }}>Q block rows</div><Select w={60} v="4" /></div>
        <div style={{ width: 150 }}><div style={{ ...muted, lineHeight: "20px" }}>HBM latency: 10</div>
          <div style={{ height: 28, display: "flex", alignItems: "center" }}><div style={{ height: 4, width: "100%", borderRadius: 2, background: C.line, position: "relative" }}><div style={{ position: "absolute", left: "15%", top: -5, width: 14, height: 14, borderRadius: 7, background: C.accent }} /></div></div></div>
      </div>
      <div style={{ ...panel, borderRadius: 4, padding: 12, marginTop: 24, height: 44, boxSizing: "border-box", fontSize: 14, lineHeight: "20px" }}>▸ Edit Q, K and V (preset: toy_sentence)</div>
      <div style={{ display: "flex", gap: 8, marginTop: 24, fontSize: 14 }}>
        <Btn w={64} active={btn === "Start"}>Start</Btn><Btn w={72} disabled>◀ Back</Btn><Btn w={72} active={btn === "Step"}>Step ▶</Btn>
        <Btn w={104} active={btn === "Run to end"} disabled={finished}>Run to end</Btn><Btn w={64}>Reset</Btn>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginTop: 24, fontSize: 14 }}>
        <Card label="Clock">{s.clock}</Card>
        <Card label="Controller" mono>{s.clock === 0 ? "not started" : s.state}</Card>
        <Card label="Tiles / K-V loads">{s.tiles} / {s.kv}</Card>
        <Card label="Expected tiles">{EXPECTED.tiles}</Card>
      </div>
      <section style={{ ...panel, borderRadius: 4, padding: 12, marginTop: 24, height: 130, boxSizing: "border-box" }}>
        <h2 style={{ margin: "0 0 8px", fontSize: 14, fontWeight: 600, lineHeight: "20px" }}>HBM traffic</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <HbmBar label="Read" value={s.rd} total={EXPECTED.rd} color={C.accent} />
          <HbmBar label="Write" value={s.wr} total={EXPECTED.wr} color={C.ok} />
        </div>
      </section>
      <div style={{ marginTop: 24 }}><Tabs tab={tab} /></div>
      <div style={{ marginTop: 24 }}>
        {tab === "schedule" && <div style={{ display: "flex", flexDirection: "column", gap: 24 }}><Schedule s={s} pulse={pulse} /><KV s={s} /></div>}
        {tab === "tile" && <TileDataflow s={s} glow={glow} />}
        {tab === "compare" && <Compare p={cmpP} sweepP={sweepP} shown={cmpShown} btn={cmpBtn} />}
        {tab === "softmax" && <p style={{ ...muted, fontSize: 14 }}>Online softmax cell view.</p>}
      </div>
    </div>
  );
};
