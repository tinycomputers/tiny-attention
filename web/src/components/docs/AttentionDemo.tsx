// Float attention for one row of a preset: the probabilities over the keys. No engine needed.
import { useMemo, useState } from "react";
import { PRESETS } from "@/lib/presets";
import { attention } from "@/lib/reference";

export default function AttentionDemo() {
  const [preset, setPreset] = useState(0);
  const p = PRESETS[preset] ?? PRESETS[0];
  const n = p?.q.length ?? 0;
  const [row, setRow] = useState(0);
  const labels = p?.labels ?? [];

  const probs = useMemo(() => (p ? attention(p.q, p.k, p.v, p.causal).probs : []), [p]);
  const weights = probs[row] ?? [];

  return (
    <div className="demo space-y-4 text-sm">
      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col">
          <span style={{ color: "var(--muted)" }}>Preset</span>
          <select className="rounded px-2 py-1" style={{ background: "var(--bg)" }} value={preset}
            onChange={(e) => { setPreset(Number(e.target.value)); setRow(0); }}>
            {PRESETS.map((x, i) => <option key={x.name} value={i}>{x.name}</option>)}
          </select>
        </label>
        <label className="flex flex-col">
          <span style={{ color: "var(--muted)" }}>Token that asks (query)</span>
          <select className="rounded px-2 py-1" style={{ background: "var(--bg)" }} value={row}
            onChange={(e) => setRow(Number(e.target.value))}>
            {Array.from({ length: n }, (_, i) => <option key={i} value={i}>{i}: {labels[i] ?? ""}</option>)}
          </select>
        </label>
      </div>
      <div className="space-y-1">
        {weights.map((w, j) => (
          <div key={j} className="flex items-center gap-2">
            <span className="w-28 truncate text-right" style={{ color: "var(--muted)" }}>{labels[j] ?? j}</span>
            <div className="h-4 flex-1 rounded" style={{ background: "var(--bg)" }}>
              <div className="h-4 rounded" style={{ width: `${Math.max(0, w) * 100}%`, background: "var(--accent)" }} />
            </div>
            <span className="w-14 tabular-nums">{(w * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <p className="note text-xs" style={{ color: "var(--muted)" }}>
        Each bar is one key&apos;s share of the query&apos;s attention. The shares sum to 100%. Causal presets give later tokens no weight.
      </p>
    </div>
  );
}
