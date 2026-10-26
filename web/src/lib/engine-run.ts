// Whole runs of the engine, for the comparison view and the N sweep. Each run uses its own instance, freed afterwards.
import { quantize } from "@/lib/quantize";
import type { EngineModule } from "@/lib/state-schema";

const MAX_CYCLES = 400_000;

export interface RunConfig {
  n: number;
  qrows: number;
  causal: boolean;
  naive: boolean;
  overlap: boolean;
  latency: number;
}

export interface RunSummary {
  out: number[][]; // n rows x 4, dequantized (float units)
  hbmRd: number;
  hbmWr: number;
  cycles: number;
  tiles: number;
  kvLoads: number;
}

/** Run to DONE and read everything. The counters are final one edge after DONE, so one more edge is taken. */
export function runEngine(mod: EngineModule, q: number[][], k: number[][], v: number[][], cfg: RunConfig): RunSummary {
  const qz = quantize(q, k, v);
  const sim = new mod.TinyAttnSim();
  try {
    sim.configure(cfg.n, cfg.qrows, cfg.causal, cfg.naive ? 1 : 0, cfg.overlap && !cfg.naive, cfg.latency, qz.cfgScale);
    sim.loadQKV(qz.q, qz.k, qz.v);
    sim.start();
    if (sim.runToDone(MAX_CYCLES) < 0) throw new Error("engine did not finish");
    const last = sim.step();
    const flat = sim.readOutput();
    const out: number[][] = [];
    for (let i = 0; i < cfg.n; i++) {
      out.push([0, 1, 2, 3].map((c) => ((flat[i * 4 + c] ?? 0) / 256) * qz.sv));
    }
    return { out, hbmRd: last.hbmRdBytes, hbmWr: last.hbmWrBytes, cycles: last.cycles, tiles: last.tiles, kvLoads: last.kvLoads };
  } finally {
    sim.delete();
  }
}

/** Deterministic inputs for sweeps (not the preset): uniform Q and K in [-2, 2), V in [-1, 1). */
export function seededInputs(n: number, seed: number): { q: number[][]; k: number[][]; v: number[][] } {
  let s = seed >>> 0;
  const next = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const mat = (amp: number) => Array.from({ length: n }, () => [0, 1, 2, 3].map(() => (next() * 2 - 1) * amp));
  return { q: mat(2), k: mat(2), v: mat(1) };
}

export interface SweepRow {
  n: number;
  flashBytes: number;
  naiveBytes: number;
  flashCycles: number;
  naiveCycles: number;
}

/** HBM bytes and cycles for N = 4, 8, ..., 32 under both schedules, on seeded inputs. */
export function sweep(mod: EngineModule, qrows: number, causal: boolean, latency: number, seed: number): SweepRow[] {
  const rows: SweepRow[] = [];
  for (let n = 4; n <= 32; n += 4) {
    const { q, k, v } = seededInputs(n, seed + n);
    const base = { n, qrows: Math.min(qrows, n), causal, overlap: false, latency };
    const f = runEngine(mod, q, k, v, { ...base, naive: false });
    const nv = runEngine(mod, q, k, v, { ...base, qrows: 4, naive: true });
    rows.push({ n, flashBytes: f.hbmRd + f.hbmWr, naiveBytes: nv.hbmRd + nv.hbmWr, flashCycles: f.cycles, naiveCycles: nv.cycles });
  }
  return rows;
}
