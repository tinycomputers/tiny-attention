import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AttentionHeatmap from "@/components/AttentionHeatmap";
import CompareView from "@/components/CompareView";
import LevelTabs, { type LevelDef } from "@/components/LevelTabs";
import SoftmaxCellView from "@/components/SoftmaxCellView";
import TileView from "@/components/TileView";
import HbmMeter from "@/components/HbmMeter";
import MatrixEditor from "@/components/MatrixEditor";
import SramPanel from "@/components/SramPanel";
import { loadEngine } from "@/lib/engine";
import { PRESETS } from "@/lib/presets";
import { quantize } from "@/lib/quantize";
import type { TileSample } from "@/lib/softmax-cell";
import { attention } from "@/lib/reference";
import { flashTileOrder, naiveTileOrder, type TileRef } from "@/lib/schedule";
import { TRAFFIC } from "@/lib/traffic-table";
import {
  FLASH_STATES,
  NAIVE_STATES,
  type EngineInstance,
  type Datapath,
  type EngineModule,
  type KVBuffers,
  type Snapshot,
  type TileState,
} from "@/lib/state-schema";

const MAX_STEPS = 400_000;

type Level = "softmax" | "tile" | "schedule" | "compare";

const LEVELS: LevelDef<Level>[] = [
  { id: "softmax", label: "L1 Softmax cell" },
  { id: "tile", label: "L2 Tile dataflow" },
  { id: "schedule", label: "L3 Schedule" },
  { id: "compare", label: "L4 Naive vs Flash" },
];

interface Settings {
  preset: number;
  mode: 0 | 1;
  causal: boolean;
  overlap: boolean;
  qrows: number;
  latency: number;
}

interface Inputs {
  q: number[][];
  k: number[][];
  v: number[][];
}

interface Result {
  out: number[][];
  maxErr: number;
  probs: number[][];
  probsSource: "engine" | "float";
}

const panel = { background: "var(--panel)" } as const;
const muted = { color: "var(--muted)" } as const;

function matrix(flat: number[], rows: number, cols: number): number[][] {
  const m: number[][] = [];
  for (let r = 0; r < rows; r++) m.push(flat.slice(r * cols, (r + 1) * cols));
  return m;
}

function maxAbsDiff(a: number[][], b: number[][]): number {
  let worst = 0;
  a.forEach((row, i) => row.forEach((v, c) => (worst = Math.max(worst, Math.abs(v - (b[i]?.[c] ?? 0))))));
  return worst;
}

export default function Visualizer() {
  const [mod, setMod] = useState<EngineModule | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings>({
    preset: 0,
    mode: 0,
    causal: PRESETS[0]?.causal ?? false,
    overlap: false,
    qrows: 4,
    latency: 10,
  });
  // Edited Q/K/V; null means the preset's values.
  const [edits, setEdits] = useState<Inputs | null>(null);
  const [stepCount, setStepCount] = useState(0);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [tile, setTile] = useState<TileState | null>(null);
  const [kv, setKv] = useState<KVBuffers | null>(null);
  const [finished, setFinished] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const simRef = useRef<EngineInstance | null>(null);
  const [level, setLevel] = useState<Level>("schedule");
  const [datapath, setDatapath] = useState<Datapath | null>(null);
  const [samples, setSamples] = useState<TileSample[]>([]);
  // completed tiles of the current run, in order (filled at each tile_done pulse; cleared when a run restarts)
  const historyRef = useRef<TileSample[]>([]);
  const orderRef = useRef<TileRef[]>([]);
  // datapath wires at the last busy clock of the run so far (between tiles the wires are idle)
  const lastBusyDatapath = useRef<Datapath | null>(null);

  const preset = PRESETS[settings.preset] ?? PRESETS[0];
  const n = preset?.q.length ?? 4;
  const labels: string[] = preset?.labels ?? [];
  const inputs: Inputs = useMemo(
    () => edits ?? { q: preset?.q ?? [], k: preset?.k ?? [], v: preset?.v ?? [] },
    [edits, preset],
  );
  const qrowOptions = [4, 8, 16, 32].filter((q) => q <= n);

  const reset = useCallback(() => {
    simRef.current?.delete();
    simRef.current = null;
    setStepCount(0);
    setSnap(null);
    setTile(null);
    setKv(null);
    setDatapath(null);
    setSamples([]);
    historyRef.current = [];
    lastBusyDatapath.current = null;
    setFinished(false);
    setResult(null);
  }, []);

  useEffect(() => {
    loadEngine().then(setMod).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    return () => simRef.current?.delete();
  }, []);

  const order: TileRef[] = useMemo(
    () =>
      settings.mode === 0
        ? flashTileOrder(n, settings.qrows, settings.causal)
        : naiveTileOrder(n, settings.causal),
    [n, settings.mode, settings.qrows, settings.causal],
  );

  orderRef.current = order;

  /** Record a completed tile: its global row group and K/V tile, and the engine's row state at that point. */
  const observe = useCallback((sim: EngineInstance, s: Snapshot) => {
    if (s.busy) lastBusyDatapath.current = sim.getDatapath();
    if (!s.tileDone) return;
    const ref = orderRef.current[s.tiles - 1];
    if (!ref) return;
    const t = sim.getTile();
    historyRef.current.push({ qi: ref.qi, kj: ref.kj, z: t.z, m: t.m, l: t.l, alpha: t.alpha });
  }, []);

  /** Advance one clock. The DONE edge is followed by one finalizing edge (counters are final after it). */
  const advance = useCallback(
    (sim: EngineInstance, state: { count: number; finished: boolean }) => {
      if (state.finished) return null;
      let s = sim.step();
      state.count++;
      observe(sim, s);
      if (s.done) {
        s = sim.step();
        state.count++;
        observe(sim, s);
        state.finished = true;
      }
      return s;
    },
    [observe],
  );

  /** Publish the engine state to the view. Builds the result once the run has finished. */
  const commit = useCallback(
    (sim: EngineInstance, count: number, done: boolean, last: Snapshot) => {
      setStepCount(count);
      setSnap(last);
      setFinished(done);
      setTile(sim.getTile());
      setKv(sim.getKVBuffers());
      setDatapath(lastBusyDatapath.current);
      setSamples(historyRef.current.slice());
      if (!done) {
        setResult(null);
        return;
      }
      const qz = quantize(inputs.q, inputs.k, inputs.v);
      const out = matrix(sim.readOutput(), n, 4).map((row) => row.map((v) => (v / 256) * qz.sv));
      const ref = attention(inputs.q, inputs.k, inputs.v, settings.causal);
      const naive = settings.mode === 1;
      setResult({
        out,
        maxErr: maxAbsDiff(out, ref.out),
        probs: naive ? matrix(sim.readProbabilities(), n, n).map((row) => row.map((v) => v / 256)) : ref.probs,
        probsSource: naive ? "engine" : "float",
      });
    },
    [inputs, n, settings],
  );

  /** Rebuild the run from reset and advance `target` clocks. Used by Start and by step-back. */
  const seek = useCallback(
    (target: number) => {
      if (!mod) return;
      simRef.current?.delete();
      historyRef.current = [];
      lastBusyDatapath.current = null;
      const qz = quantize(inputs.q, inputs.k, inputs.v);
      const sim = new mod.TinyAttnSim();
      sim.configure(n, settings.qrows, settings.causal, settings.mode, settings.overlap, settings.latency, qz.cfgScale);
      sim.loadQKV(qz.q, qz.k, qz.v);
      sim.start();
      simRef.current = sim;

      const state = { count: 0, finished: false };
      let last: Snapshot = sim.snapshot();
      while (state.count < target && !state.finished) {
        last = advance(sim, state) ?? last;
      }
      commit(sim, state.count, state.finished, last);
    },
    [mod, inputs, n, settings, advance, commit],
  );

  const stepOnce = () => {
    const sim = simRef.current;
    if (!sim) return seek(1);
    const state = { count: stepCount, finished };
    const s = advance(sim, state);
    if (!s) return;
    commit(sim, state.count, state.finished, s);
  };

  const runToEnd = () => {
    const sim = simRef.current;
    if (!sim || finished) return;
    const state = { count: stepCount, finished };
    let last: Snapshot | null = null;
    while (!state.finished && state.count < MAX_STEPS) last = advance(sim, state) ?? last;
    commit(sim, state.count, state.finished, last ?? sim.snapshot());
  };

  const setPreset = (i: number) => {
    reset();
    setEdits(null);
    setSettings((s) => ({
      ...s,
      preset: i,
      causal: PRESETS[i]?.causal ?? false,
      qrows: Math.min(s.qrows, PRESETS[i]?.q.length ?? 4),
    }));
  };

  const editValue = (name: keyof Inputs, row: number, col: number, value: number) => {
    reset();
    setEdits((prev) => {
      const base = prev ?? inputs;
      const copy = base[name].map((r) => r.slice());
      const target = copy[row];
      if (target) target[col] = value;
      return { ...base, [name]: copy };
    });
  };

  // ---- derived view
  const nb = n / 4;
  const tilesStarted = snap?.tiles ?? 0;
  const orderIndex = new Map<string, number>();
  order.forEach((t, i) => orderIndex.set(`${t.qi},${t.kj}`, i));
  const stateNames = settings.mode === 0 ? FLASH_STATES : NAIVE_STATES;
  const stateName = snap ? (stateNames[snap.ctrlState] ?? `#${snap.ctrlState}`) : "not started";
  const qblock = settings.mode === 1 ? 4 : settings.qrows;
  const expected =
    TRAFFIC[`${settings.mode === 1 ? "NAIVE" : "FLASH"}:${n}:${settings.causal ? 1 : 0}:${qblock}`] ?? null;

  const cellClass = (qi: number, kj: number) => {
    if (settings.causal && kj > qi) return { bg: "transparent", label: "masked" };
    const idx = orderIndex.get(`${qi},${kj}`);
    if (idx === undefined) return { bg: "transparent", label: "" };
    if (idx === tilesStarted - 1 && snap?.busy) return { bg: "var(--accent)", label: "in flight" };
    if (idx < tilesStarted) return { bg: "var(--ok)", label: "done" };
    return { bg: "var(--panel)", label: "" };
  };


  if (error) return <p style={{ color: "var(--err)" }}>{error}</p>;
  if (!mod) return <p style={muted}>Loading engine…</p>;

  return (
    <div className="space-y-6">
      {/* controls */}
      <div className="flex flex-wrap items-end gap-3 text-sm">
        <label className="flex flex-col">
          <span style={muted}>Preset</span>
          <select className="rounded px-2 py-1" style={panel} value={settings.preset}
            onChange={(e) => setPreset(Number(e.target.value))}>
            {PRESETS.map((p, i) => (
              <option key={p.name} value={i}>{p.name} (N={p.q.length})</option>
            ))}
          </select>
        </label>
        <label className="flex flex-col">
          <span style={muted}>Schedule</span>
          <select className="rounded px-2 py-1" style={panel} value={settings.mode}
            onChange={(e) => { reset(); setSettings((s) => ({ ...s, mode: Number(e.target.value) as 0 | 1 })); }}>
            <option value={0}>FLASH</option>
            <option value={1}>NAIVE</option>
          </select>
        </label>
        <label className="flex items-center gap-1">
          <input type="checkbox" checked={settings.causal}
            onChange={(e) => { reset(); setSettings((s) => ({ ...s, causal: e.target.checked })); }} />
          <span>Causal</span>
        </label>
        {settings.mode === 0 && (
          <>
            <label className="flex items-center gap-1">
              <input type="checkbox" checked={settings.overlap}
                onChange={(e) => { reset(); setSettings((s) => ({ ...s, overlap: e.target.checked })); }} />
              <span>Overlap</span>
            </label>
            <label className="flex flex-col">
              <span style={muted}>Q block rows</span>
              <select className="rounded px-2 py-1" style={panel} value={settings.qrows}
                onChange={(e) => { reset(); setSettings((s) => ({ ...s, qrows: Number(e.target.value) })); }}>
                {qrowOptions.map((q) => <option key={q} value={q}>{q}</option>)}
              </select>
            </label>
          </>
        )}
        <label className="flex flex-col">
          <span style={muted}>HBM latency: {settings.latency}</span>
          <input type="range" min={0} max={64} value={settings.latency}
            onChange={(e) => { reset(); setSettings((s) => ({ ...s, latency: Number(e.target.value) })); }} />
        </label>
      </div>

      <details className="rounded p-3 text-sm" style={panel}>
        <summary className="cursor-pointer">
          Edit Q, K and V {edits ? "(edited)" : `(preset: ${preset?.name ?? ""})`}
        </summary>
        <div className="mt-3 space-y-4">
          <p style={muted} className="text-xs">
            Values are floats; the browser quantizes them to INT8 the same way as the golden model. Editing resets the run.
          </p>
          <MatrixEditor title="Q" values={inputs.q} onChange={(r, c, x) => editValue("q", r, c, x)} />
          <MatrixEditor title="K" values={inputs.k} onChange={(r, c, x) => editValue("k", r, c, x)} />
          <MatrixEditor title="V" values={inputs.v} onChange={(r, c, x) => editValue("v", r, c, x)} />
          <button className="rounded px-3 py-1" style={panel} disabled={!edits}
            onClick={() => { reset(); setEdits(null); }}>Reset to preset</button>
        </div>
      </details>

      <div className="flex flex-wrap gap-2 text-sm">
        <button className="rounded px-3 py-1" style={panel} onClick={() => seek(0)}>Start</button>
        <button className="rounded px-3 py-1" style={panel} disabled={stepCount === 0}
          onClick={() => seek(Math.max(0, stepCount - 1))}>◀ Back</button>
        <button className="rounded px-3 py-1" style={panel} disabled={!simRef.current || finished}
          onClick={stepOnce}>Step ▶</button>
        <button className="rounded px-3 py-1" style={panel} disabled={!simRef.current || finished}
          onClick={runToEnd}>Run to end</button>
        <button className="rounded px-3 py-1" style={panel} onClick={reset}>Reset</button>
      </div>

      {/* status */}
      <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div className="rounded p-3" style={panel}>
          <div style={muted}>Clock</div><div data-testid="clock" className="text-lg tabular-nums">{stepCount}</div>
        </div>
        <div className="rounded p-3" style={panel}>
          <div style={muted}>Controller</div><div className="font-mono">{stateName}</div>
        </div>
        <div className="rounded p-3" style={panel}>
          <div style={muted}>Tiles / K-V loads</div>
          <div className="tabular-nums">{snap?.tiles ?? 0} / {snap?.kvLoads ?? 0}</div>
        </div>
        <div className="rounded p-3" style={panel}>
          <div style={muted}>Expected tiles</div>
          <div className="tabular-nums">{expected?.tiles ?? "-"}</div>
        </div>
      </div>

      <section className="rounded p-3" style={panel}>
        <h2 className="mb-2 text-sm font-semibold">HBM traffic</h2>
        <HbmMeter read={snap?.hbmRdBytes ?? 0} write={snap?.hbmWrBytes ?? 0} expected={expected} />
      </section>

      <LevelTabs levels={LEVELS} value={level} onChange={setLevel} />

      <div role="tabpanel" id={`panel-${level}`} aria-labelledby={`tab-${level}`} className="space-y-6">
        {level === "schedule" && (
          <>
            <div className="grid gap-6 lg:grid-cols-2">
        {/* schedule */}
        <section>
          <h2 className="mb-2 text-sm font-semibold">
            {settings.mode === 0 ? "Tile schedule (row groups × K/V tiles)" : "Tile schedule, pass 1 (Z tiles)"}
          </h2>
          <div className="inline-grid gap-1" style={{ gridTemplateColumns: `auto repeat(${nb}, 2.25rem)` }}>
            <div />
            {Array.from({ length: nb }, (_, kj) => (
              <div key={`h${kj}`} className="text-center text-xs" style={muted}>kv{kj}</div>
            ))}
            {Array.from({ length: nb }, (_, qi) => (
              <div key={`r${qi}`} className="contents">
                <div className="pr-2 text-right text-xs" style={muted}>q{qi}</div>
                {Array.from({ length: nb }, (_, kj) => {
                  const c = cellClass(qi, kj);
                  return (
                    <div key={`${qi}-${kj}`} title={c.label} className="flex h-9 items-center justify-center rounded text-[10px]"
                      style={{ background: c.bg, border: "1px solid var(--muted)", opacity: c.label === "masked" ? 0.25 : 1 }}>
                      {c.label === "in flight" ? "●" : ""}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          <p className="mt-2 text-xs" style={muted}>
            Green: finished. Blue: in flight. Faded: masked by the causal mask.
            {settings.mode === 1 ? " NAIVE passes 2 to 4 reuse these tiles." : ""}
          </p>
        </section>
            </div>

      <section>
        <h2 className="mb-2 text-sm font-semibold">On-chip K/V buffers</h2>
        <SramPanel kv={kv} />
      </section>
          </>
        )}
        {level === "softmax" && (
          <SoftmaxCellView samples={samples} q={inputs.q} k={inputs.k} causal={settings.causal} labels={labels}
            naive={settings.mode === 1} />
        )}
        {level === "tile" && <TileView datapath={datapath} tile={tile} naive={settings.mode === 1} />}
        {level === "compare" && (
          <CompareView mod={mod} q={inputs.q} k={inputs.k} v={inputs.v} causal={settings.causal}
            qrows={settings.qrows} latency={settings.latency} overlap={settings.overlap} />
        )}
      </div>

      {/* result */}
      {result && (
        <section className="space-y-4">
          <h2 className="text-sm font-semibold">Output</h2>
          <p className="text-sm" style={muted}>
            Max |engine − float attention| = {result.maxErr.toFixed(4)} (INT8 quantization of Q, K and V dominates).
          </p>
          <Grid title="O (dequantized)" values={result.out.flat()} cols={4} />
          <AttentionHeatmap probs={result.probs} labels={labels} source={result.probsSource} />
        </section>
      )}
    </div>
  );
}

function Grid({ title, values, cols }: { title: string; values: number[]; cols: number }) {
  const rows: number[][] = [];
  for (let r = 0; r < values.length / cols; r++) rows.push(values.slice(r * cols, (r + 1) * cols));
  return (
    <div>
      <div style={muted}>{title}</div>
      <div className="overflow-x-auto">
        <div className="inline-grid gap-x-3 font-mono" style={{ gridTemplateColumns: `repeat(${cols}, auto)` }}>
          {rows.flatMap((row, i) =>
            row.map((v, j) => <span key={`${i}-${j}`}>{Number.isInteger(v) ? v : v.toFixed(4)}</span>),
          )}
        </div>
      </div>
    </div>
  );
}
