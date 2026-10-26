// HBM traffic for FLASH and NAIVE from the golden traffic model (web/src/lib/traffic-table.ts, generated).
// The cocotb end-to-end test checks these formulas against the RTL byte counters for every N, causal setting and block size.
import { useState } from "react";
import { TRAFFIC } from "@/lib/traffic-table";

const NS = [4, 8, 12, 16, 20, 24, 28, 32];

export default function TrafficDemo() {
  const [n, setN] = useState(32);
  const [causal, setCausal] = useState(false);
  const [qrows, setQrows] = useState(4);
  const c = causal ? 1 : 0;
  const flash = TRAFFIC[`FLASH:${n}:${c}:${qrows}`];
  const naive = TRAFFIC[`NAIVE:${n}:${c}:4`];
  const flashTotal = flash ? flash.rd + flash.wr : 0;
  const naiveTotal = naive ? naive.rd + naive.wr : 0;
  const max = Math.max(flashTotal, naiveTotal, 1);

  return (
    <div className="demo space-y-4 text-sm">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col">
          <span style={{ color: "var(--muted)" }}>N</span>
          <select className="rounded px-2 py-1" style={{ background: "var(--bg)" }} value={n}
            onChange={(e) => setN(Number(e.target.value))}>
            {NS.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="flex flex-col">
          <span style={{ color: "var(--muted)" }}>Q block rows (FLASH)</span>
          <select className="rounded px-2 py-1" style={{ background: "var(--bg)" }} value={qrows}
            onChange={(e) => setQrows(Number(e.target.value))}>
            {[4, 8, 16, 32].filter((q) => q <= n).map((q) => <option key={q} value={q}>{q}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-2 pb-1">
          <input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} />
          <span>Causal</span>
        </label>
      </div>

      {[
        { name: "NAIVE (N×N scores in HBM)", total: naiveTotal, color: "var(--err)" },
        { name: `FLASH (Q block ${qrows} rows)`, total: flashTotal, color: "var(--accent)" },
      ].map((row) => (
        <div key={row.name} className="space-y-1">
          <div className="flex justify-between tabular-nums">
            <span>{row.name}</span>
            <span>{row.total} B</span>
          </div>
          <div className="h-3 rounded" style={{ background: "var(--bg)" }}>
            <div className="h-3 rounded" style={{ width: `${(row.total / max) * 100}%`, background: row.color }} />
          </div>
        </div>
      ))}

      <p style={{ color: "var(--muted)" }}>
        FLASH moves {flashTotal ? (naiveTotal / flashTotal).toFixed(1) : "-"}× fewer bytes than NAIVE here.
        Tiles computed: FLASH {flash?.tiles ?? "-"}, NAIVE {naive?.tiles ?? "-"}.
      </p>
    </div>
  );
}
