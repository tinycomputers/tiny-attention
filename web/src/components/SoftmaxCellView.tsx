// L1: one row lane of the online softmax. Each K/V tile adds four logits; the running max m rises, the rescale α scales
// the old sum down, and the sum l grows. Values are the engine's, read at each tile completion. The reference column is
// the float softmax of the same INT8 inputs (rowReference), so the differences are fixed-point rounding only.
import { useMemo, useState } from "react";
import { rowReference, rowSteps, type TileSample } from "@/lib/softmax-cell";

interface Props {
  samples: TileSample[];
  q: number[][];
  k: number[][];
  causal: boolean;
  labels: string[];
  naive: boolean;
}

const muted = { color: "var(--muted)" } as const;
const fmt = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "-");

export default function SoftmaxCellView({ samples, q, k, causal, labels, naive }: Props) {
  const [row, setRow] = useState(0);
  const n = q.length;
  const steps = useMemo(() => rowSteps(samples, row), [samples, row]);
  const ref = useMemo(() => rowReference(q, k, row, causal), [q, k, row, causal]);

  if (naive) {
    return (
      <p className="text-sm" style={muted}>
        NAIVE has no running max: the softmax of every row is computed in three passes over HBM. Switch to FLASH to see the
        online cell.
      </p>
    );
  }
  if (steps.length === 0) {
    return <p className="text-sm" style={muted}>Run the engine (Start, then Step or Run to end) to fill this view.</p>;
  }

  const last = steps[steps.length - 1];

  // polyline of m and l across the row's tiles, each scaled to its own range
  const W = 320;
  const H = 110;
  const pts = (vals: number[]) => {
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const span = hi - lo || 1;
    return vals.map((v, i) => [
      (i / Math.max(1, vals.length - 1)) * (W - 20) + 10,
      H - 10 - ((v - lo) / span) * (H - 20),
    ] as const);
  };
  const polyline = (vals: number[]) => pts(vals).map(([x, y]) => `${x},${y}`).join(" ");
  const mVals = steps.map((s) => s.m);
  const lVals = steps.map((s) => s.l);

  return (
    <div className="space-y-4 text-sm">
      <label className="flex items-center gap-2">
        <span style={muted}>Token (row)</span>
        <select className="rounded px-2 py-1" style={{ background: "var(--panel)" }} value={row}
          onChange={(e) => setRow(Number(e.target.value))}>
          {Array.from({ length: n }, (_, i) => (
            <option key={i} value={i}>{i}: {labels[i] ?? ""}</option>
          ))}
        </select>
      </label>

      <div className="overflow-x-auto">
        <table className="w-full text-xs tabular-nums">
          <thead>
            <tr style={muted} className="text-left">
              <th className="pr-2">K/V tile</th>
              <th className="pr-2">logits, engine (log2)</th>
              <th className="pr-2">logits, float</th>
              <th className="pr-2">m</th>
              <th className="pr-2">α</th>
              <th className="pr-2">l</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s, i) => (
              <tr key={i} className="border-t" style={{ borderColor: "var(--line)" }}>
                <td className="pr-2">kv{s.kj}</td>
                <td className="pr-2 font-mono">
                  {s.z.map((x) => (x < -127 ? "masked" : fmt(x))).join("  ")}
                </td>
                <td className="pr-2 font-mono">
                  {[0, 1, 2, 3].map((c) => fmt(ref.z[s.kj * 4 + c] ?? Number.NaN)).join("  ")}
                </td>
                <td className="pr-2">{fmt(s.m)}</td>
                <td className="pr-2">{fmt(s.alpha, 3)}</td>
                <td className="pr-2">{fmt(s.l, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md" role="img" aria-label="running max and sum across tiles">
        <polyline points={polyline(mVals)} fill="none" stroke="var(--accent)" strokeWidth={2} />
        <polyline points={polyline(lVals)} fill="none" stroke="var(--ok)" strokeWidth={2} strokeDasharray="4 3" />
        {pts(mVals).map(([x, y], i) => <circle key={`m${i}`} cx={x} cy={y} r={3} fill="var(--accent)" />)}
        {pts(lVals).map(([x, y], i) => <circle key={`l${i}`} cx={x} cy={y} r={3} fill="var(--ok)" />)}
      </svg>
      <div className="flex gap-4 text-xs" style={muted}>
        <span><span style={{ color: "var(--accent)" }}>━</span> m (running max, log2)</span>
        <span><span style={{ color: "var(--ok)" }}>┅</span> l (running sum, scaled per line)</span>
      </div>

      <div className="rounded p-3" style={{ background: "var(--panel)" }}>
        <div className="font-semibold">After all tiles</div>
        <div className="mt-1 grid grid-cols-2 gap-x-6 gap-y-1 tabular-nums sm:grid-cols-3">
          <span style={muted}>m engine / float</span>
          <span>{fmt(last?.m ?? Number.NaN, 3)} / {fmt(ref.m, 3)}</span>
          <span style={muted}>l engine / float</span>
          <span>{fmt(last?.l ?? Number.NaN, 3)} / {fmt(ref.l, 3)}</span>
          <span style={muted}>|Δm|, |Δl|</span>
          <span>{fmt(Math.abs((last?.m ?? 0) - ref.m), 4)}, {fmt(Math.abs((last?.l ?? 0) - ref.l), 4)}</span>
        </div>
        <p className="mt-2 text-xs" style={muted}>
          Each logit is split into an integer shift and a LUT fraction inside exp2 (`2^z = 2^⌊z⌋ · 2^frac(z)`). The float
          column is the same INT8 inputs in full precision.
        </p>
      </div>
    </div>
  );
}
