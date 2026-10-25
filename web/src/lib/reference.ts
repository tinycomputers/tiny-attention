// Float64 attention, the reference the engine output is compared with (sim/golden/float_ref.py).

export function attention(q: number[][], k: number[][], v: number[][], causal: boolean): {
  out: number[][];
  probs: number[][];
} {
  const n = q.length;
  const d = q[0]?.length ?? 0;
  const probs: number[][] = [];
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    const s: number[] = [];
    for (let j = 0; j < n; j++) {
      if (causal && j > i) {
        s.push(-Infinity);
        continue;
      }
      let dot = 0;
      for (let c = 0; c < d; c++) dot += (q[i]?.[c] ?? 0) * (k[j]?.[c] ?? 0);
      s.push(dot / Math.sqrt(d));
    }
    const mx = Math.max(...s);
    const e = s.map((x) => (x === -Infinity ? 0 : Math.exp(x - mx)));
    const sum = e.reduce((a, b) => a + b, 0);
    const p = e.map((x) => x / sum);
    probs.push(p);
    const row: number[] = new Array(d).fill(0);
    for (let j = 0; j < n; j++) for (let c = 0; c < d; c++) row[c] = (row[c] ?? 0) + (p[j] ?? 0) * (v[j]?.[c] ?? 0);
    out.push(row);
  }
  return { out, probs };
}
