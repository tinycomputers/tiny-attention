// Host-side quantization, a mirror of sim/golden/fixed.py quantize(): symmetric per-tensor INT8 and the folded
// cfg_scale (UQ4.20, 24 bits). Checked against wasm/fixtures/quantize_cases.json in quantize.test.ts.

export const LOG2E = Math.log2(Math.E);
export const D = 4;

/** numpy.rint / Python round: halves go to the even neighbour. */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export interface Quantized {
  q: number[]; // int8, row-major
  k: number[];
  v: number[];
  sq: number;
  sk: number;
  sv: number;
  cfgScale: number; // 24-bit UQ4.20
}

function quantizeOne(x: number[][]): { values: number[]; scale: number } {
  let maxAbs = 0;
  for (const row of x) for (const value of row) maxAbs = Math.max(maxAbs, Math.abs(value));
  const scale = maxAbs / 127 || 1.0;
  const values: number[] = [];
  for (const row of x) {
    for (const value of row) {
      values.push(Math.min(127, Math.max(-127, roundHalfEven(value / scale))));
    }
  }
  return { values, scale };
}

export function quantize(q: number[][], k: number[][], v: number[][]): Quantized {
  const Q = quantizeOne(q);
  const K = quantizeOne(k);
  const V = quantizeOne(v);
  const cfgScale = roundHalfEven(Q.scale * K.scale * LOG2E / Math.sqrt(D) * 2 ** 20);
  if (cfgScale < 0 || cfgScale >= 2 ** 24) throw new Error(`cfg_scale ${cfgScale} does not fit 24 bits`);
  return { q: Q.values, k: K.values, v: V.values, sq: Q.scale, sk: K.scale, sv: V.scale, cfgScale };
}
