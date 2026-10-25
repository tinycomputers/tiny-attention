import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { loadEngine } from "@/lib/engine";
import { PRESETS } from "@/lib/presets";
import { quantize } from "@/lib/quantize";
import { attention } from "@/lib/reference";
import { flashTileOrder, naiveTileOrder, type TileRef } from "@/lib/schedule";
import {
  FLASH_STATES,
  NAIVE_STATES,
  type EngineInstance,
  type EngineModule,
  type Snapshot,
  type TileState,
} from "@/lib/state-schema";

const MAX_STEPS = 400_000;

interface Settings {
  preset: number;
  mode: 0 | 1;
  causal: boolean;
  overlap: boolean;
  qrows: number;
  latency: number;
}

interface Result {
  out: number[][];
  maxErr: number;
  probs: number[][] | null;
}

const panel = { background: "var(--panel)" } as const;
const muted = { color: "var(--muted)" } as const;

function matrix(flat: number[], rows: number, cols: number): number[][] {
  const m: number[][] = [];
  for (let r = 0; r < rows; r++) m.push(flat.slice(r * cols, (r + 1) * cols));
  return m;
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
  const [stepCount, setStepCount] = useState(0);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [tile, setTile] = useState<TileState | null>(null);
  const [finished, setFinished] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const simRef = useRef<EngineInstance | null>(null);

  const preset = PRESETS[settings.preset] ?? PRESETS[0];
  const n = preset?.q.length ?? 4;
  const labels: string[] = preset?.labels ?? [];
  const qrowOptions = [4, 8, 16, 32].filter((q) => q <= n);

  useEffect(() => {
    loadEngine().then(setMod).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    return () => simRef.current?.delete();
  }, []);

  const order: TileRef[] = useMemo(() => {
    if (!preset) return [];
    return settings.mode === 0
      ? flashTileOrder(n, settings.qrows, settings.causal)
      : naiveTileOrder(n, settings.causal);
  }, [preset, n, settings.mode, settings.qrows, settings.causal]);

  /** Advance one clock. The DONE edge is followed by one finalizing edge (counters are final after it). */
  const advance = useCallback((sim: EngineInstance, state: { count: number; finished: boolean }) => {
    if (state.finished) return null;
    let s = sim.step();
    state.count++;
    if (s.done) {
      s = sim.step();
      state.count++;
      state.finished = true;
    }
    return s;
  }, []);

  /** Publish the state of the engine to the view. Builds the result once the run has finished. */
  const commit = useCallback(
    (sim: EngineInstance, count: number, done: boolean, last: Snapshot) => {
      setStepCount(count);
      setSnap(last);
      setFinished(done);
      setTile(sim.getTile());
      if (!done || !preset) {
        setResult(null);
        return;
      }
      const qz = quantize(preset.q, preset.k, preset.v);
      const out = matrix(sim.readOutput(), n, 4).map((row) => row.map((v) => (v / 256) * qz.sv));
      const ref = attention(preset.q, preset.k, preset.v, settings.causal).out;
      let maxErr = 0;
      out.forEach((row, i) => row.forEach((v, c) => (maxErr = Math.max(maxErr, Math.abs(v - (ref[i]?.[c] ?? 0))))));
      const probs =
        settings.mode === 1 ? matrix(sim.readProbabilities(), n, n).map((row) => row.map((v) => v / 256)) : null;
      setResult({ out, maxErr, probs });
    },
    [preset, n, settings],
  );

  /** Rebuild the run from reset and advance `target` clocks. Used by Start and by step-back. */
  const seek = useCallback(
    (target: number) => {
      if (!mod || !preset) return;
      simRef.current?.delete();
      const qz = quantize(preset.q, preset.k, preset.v);
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
    [mod, preset, n, settings, advance, commit],
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

  const reset = () => {
    simRef.current?.delete();
    simRef.current = null;
    setStepCount(0);
    setSnap(null);
    setTile(null);
    setFinished(false);
    setResult(null);
  };

  const setPreset = (i: number) => {
    reset();
    setSettings((s) => ({ ...s, preset: i, causal: PRESETS[i]?.causal ?? false, qrows: Math.min(s.qrows, PRESETS[i]?.q.length ?? 4) }));
  };

  // ---- derived view
  const nb = n / 4;
  const tilesStarted = snap?.tiles ?? 0;
  const orderIndex = new Map<string, number>();
  order.forEach((t, i) => orderIndex.set(`${t.qi},${t.kj}`, i));
  const stateNames = settings.mode === 0 ? FLASH_STATES : NAIVE_STATES;
  const stateName = snap ? (stateNames[snap.ctrlState] ?? `#${snap.ctrlState}`) : "not started";
  const cellClass = (qi: number, kj: number) => {
    if (settings.causal && kj > qi) return { bg: "transparent", label: "masked" };
    const idx = orderIndex.get(`${qi},${kj}`);
    if (idx === undefined) return { bg: "transparent", label: "" };
    if (idx < tilesStarted - 1) return { bg: "var(--ok)", label: "done" };
    if (idx === tilesStarted - 1 && snap?.busy) return { bg: "var(--accent)", label: "in flight" };
    if (idx === tilesStarted - 1) return { bg: "var(--ok)", label: "done" };
    return { bg: "var(--panel)", label: "" };
  };

  const fmt = (v: number, digits = 0) => (Number.isFinite(v) ? v.toFixed(digits) : "-");

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
          <div style={muted}>Clock</div><div className="text-lg tabular-nums">{stepCount}</div>
        </div>
        <div className="rounded p-3" style={panel}>
          <div style={muted}>Controller</div><div className="font-mono">{stateName}</div>
        </div>
        <div className="rounded p-3" style={panel}>
          <div style={muted}>HBM read / write</div>
          <div className="tabular-nums">{snap?.hbmRdBytes ?? 0} B / {snap?.hbmWrBytes ?? 0} B</div>
        </div>
        <div className="rounded p-3" style={panel}>
          <div style={muted}>Tiles / K-V loads</div>
          <div className="tabular-nums">{snap?.tiles ?? 0} / {snap?.kvLoads ?? 0}</div>
        </div>
      </div>

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

        {/* current tile */}
        <section>
          <h2 className="mb-2 text-sm font-semibold">Current tile values</h2>
          {tile && tilesStarted > 0 ? (
            <div className="space-y-3 text-xs tabular-nums">
              {settings.mode === 0 ? (
                <>
                  <p style={muted}>Row group {tile.curGrp} (rows {tile.curGrp * 4}–{tile.curGrp * 4 + 3})</p>
                  <Grid title="S = Q·Kᵀ (int)" values={tile.s} cols={4} />
                  <Grid title="Z (log2 logits)" values={tile.z} cols={4} />
                  <Grid title="P (UQ1.8)" values={tile.p} cols={4} />
                  <div className="grid grid-cols-3 gap-2">
                    <Vec title="m" values={tile.m} fmt={(v) => fmt(v)} />
                    <Vec title="l" values={tile.l} fmt={(v) => fmt(v)} />
                    <Vec title="α" values={tile.alpha} fmt={(v) => fmt(v)} />
                  </div>
                </>
              ) : (
                <p style={muted}>
                  NAIVE tiles go through HBM, not the tile core, so the per-element S, Z and P are not shown here.
                  The row max m and sum l are shown below; the final probabilities P are in the matrix after the run.
                </p>
              )}
              {settings.mode === 1 && (
                <div className="grid grid-cols-2 gap-2">
                  <Vec title="m" values={tile.m} fmt={(v) => fmt(v)} />
                  <Vec title="l" values={tile.l} fmt={(v) => fmt(v)} />
                </div>
              )}
              <Grid title="O (Q8.8 accumulator)" values={tile.o} cols={4} />
            </div>
          ) : (
            <p className="text-sm" style={muted}>Press Start, then Step.</p>
          )}
        </section>
      </div>

      {/* result */}
      {result && (
        <section className="space-y-4">
          <h2 className="text-sm font-semibold">Output</h2>
          <p className="text-sm" style={muted}>
            Max |engine − float attention| = {result.maxErr.toFixed(4)} (INT8 quantization of Q, K and V dominates).
          </p>
          <Grid title="O (dequantized)" values={result.out.flat()} cols={4} />
          {result.probs && (
            <div>
              <h3 className="mb-2 text-sm font-semibold">Attention probabilities P (from HBM, NAIVE)</h3>
              <div className="inline-grid gap-px" style={{ gridTemplateColumns: `repeat(${n}, 1rem)` }}>
                {result.probs.flatMap((row, i) =>
                  row.map((p, j) => (
                    <div key={`${i}-${j}`} title={`${labels[i] ?? i} → ${labels[j] ?? j}: ${p.toFixed(3)}`}
                      style={{ width: "1rem", height: "1rem", background: `rgba(88, 166, 255, ${Math.min(1, p * 1.6)})` }} />
                  )),
                )}
              </div>
              {labels.length === n && (
                <div className="mt-2 flex flex-wrap gap-x-3 text-[10px]" style={muted}>
                  {labels.map((l, i) => <span key={i}>{i}:{l}</span>)}
                </div>
              )}
            </div>
          )}
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
      <div className="inline-grid gap-x-3 font-mono" style={{ gridTemplateColumns: `repeat(${cols}, auto)` }}>
        {rows.flatMap((row, i) => row.map((v, j) => <span key={`${i}-${j}`}>{v}</span>))}
      </div>
    </div>
  );
}

function Vec({ title, values, fmt }: { title: string; values: number[]; fmt: (v: number) => string }) {
  return (
    <div>
      <div style={muted}>{title}</div>
      <div className="font-mono">{values.map((v) => fmt(v)).join(" ")}</div>
    </div>
  );
}
