// One softmax row, in the engine's units, and the float reference for the same INT8 inputs.
//
// The engine works in log2 units: z = S · log2(e) / sqrt(d), with S = q·k computed on INT8 values. The float reference
// here uses the same INT8 values dequantized (q_int · s_q), so the only difference left is the engine's fixed-point
// rounding, not the quantization.
import { D, LOG2E, quantize } from "@/lib/quantize";

export interface RowReference {
  z: number[]; // log2-domain logit per key; -Infinity where the causal mask removes the key
  m: number; // max over the allowed keys
  l: number; // sum of 2^(z - m) over the allowed keys
}

export function rowReference(q: number[][], k: number[][], row: number, causal: boolean): RowReference {
  const qz = quantize(q, k, q);
  const n = q.length;
  const qd = (i: number, c: number) => (qz.q[i * D + c] ?? 0) * qz.sq;
  const kd = (j: number, c: number) => (qz.k[j * D + c] ?? 0) * qz.sk;
  const z: number[] = [];
  for (let j = 0; j < n; j++) {
    if (causal && j > row) {
      z.push(-Infinity);
      continue;
    }
    let dot = 0;
    for (let c = 0; c < D; c++) dot += qd(row, c) * kd(j, c);
    z.push((dot / Math.sqrt(D)) * LOG2E);
  }
  const m = Math.max(...z.filter((x) => Number.isFinite(x)));
  const l = z.reduce((acc, x) => (Number.isFinite(x) ? acc + 2 ** (x - m) : acc), 0);
  return { z, m, l };
}

/** A completed tile as the engine reported it (recorded at the tile_done pulse). */
export interface TileSample {
  qi: number; // global row group
  kj: number; // K/V tile
  z: number[]; // 16 logits, Q8.8 log2 units (row-major: row, key column)
  m: number[]; // 4 rows, Q8.8
  l: number[]; // 4 rows, real × 2^12
  alpha: number[]; // 4 rows, UQ1.15
}

export interface RowStep {
  kj: number;
  z: number[]; // 4 logits of this tile for the row, real units
  m: number; // running max after the tile
  alpha: number; // rescale factor applied to the old sum (0..1)
  l: number; // running sum after the tile
}

/** The steps row `row` went through, in tile order. */
export function rowSteps(samples: TileSample[], row: number): RowStep[] {
  const g = Math.floor(row / 4);
  const r = row % 4;
  return samples
    .filter((s) => s.qi === g)
    .map((s) => ({
      kj: s.kj,
      z: [0, 1, 2, 3].map((c) => (s.z[r * 4 + c] ?? 0) / 256),
      m: (s.m[r] ?? 0) / 256,
      alpha: (s.alpha[r] ?? 0) / 32768,
      l: (s.l[r] ?? 0) / 4096,
    }));
}
