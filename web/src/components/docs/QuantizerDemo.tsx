// Host-side INT8 quantization of one Q row and one K row, the way the engine's inputs are prepared (quantize.ts mirrors
// sim/golden/fixed.py). Shows the INT8 values, the per-tensor scales, and the folded cfg_scale in hex.
import { useMemo, useState } from "react";
import { D, LOG2E, quantize } from "@/lib/quantize";

const fmtNum = (x: number) => x.toFixed(4);

export default function QuantizerDemo() {
  const [qRow, setQRow] = useState([1.2, -0.7, 2.4, 0.1]);
  const [kRow, setKRow] = useState([0.9, 1.8, -2.1, 0.4]);

  const result = useMemo(() => {
    const vRow = [[0.5, -0.5, 0.25, 1]];
    return quantize([qRow], [kRow], vRow);
  }, [qRow, kRow]);

  const setCell = (which: "q" | "k", i: number, value: number) => {
    const update = (row: number[]) => row.map((x, j) => (j === i ? value : x));
    if (which === "q") setQRow((r) => update(r));
    else setKRow((r) => update(r));
  };

  const dq = (ints: number[], s: number) => ints.map((x) => x * s);

  return (
    <div className="demo space-y-4 text-sm">
      {(["q", "k"] as const).map((which) => {
        const row = which === "q" ? qRow : kRow;
        const ints = which === "q" ? result.q : result.k;
        const scale = which === "q" ? result.sq : result.sk;
        return (
          <div key={which} className="space-y-2">
            <div className="font-semibold">{which === "q" ? "Q row" : "K row"}</div>
            <div className="grid grid-cols-4 gap-2">
              {row.map((x, i) => (
                <input key={i} type="number" step="0.1" value={x} aria-label={`${which} dimension ${i}`}
                  className="rounded px-2 py-1 font-mono" style={{ background: "var(--bg)" }}
                  onChange={(e) => { const v = e.target.valueAsNumber; if (Number.isFinite(v)) setCell(which, i, v); }} />
              ))}
            </div>
            <div className="grid grid-cols-4 gap-2 font-mono text-xs">
              {ints.map((x, i) => (
                <div key={i} className="rounded p-2" style={{ background: "var(--bg)" }}>
                  <div style={{ color: "var(--muted)" }}>int8 {x}</div>
                  <div>≈ {fmtNum(dq([x], scale)[0] ?? 0)}</div>
                </div>
              ))}
            </div>
            <div className="text-xs" style={{ color: "var(--muted)" }}>
              scale {scale.toFixed(6)} (int8 = round(x / scale), halves to even; step error ≤ scale / 2 = {(scale / 2).toFixed(4)})
            </div>
          </div>
        );
      })}
      <div className="rounded p-3 font-mono text-xs" style={{ background: "var(--bg)" }}>
        <div>cfg_scale = round(s_Q · s_K · log2(e) / √{D} · 2²⁰)</div>
        <div className="mt-1">
          = {result.cfgScale} = 0x{result.cfgScale.toString(16).padStart(6, "0")} (24 bits, UQ4.20; scales log2(e)≈{LOG2E.toFixed(4)})
        </div>
      </div>
    </div>
  );
}
