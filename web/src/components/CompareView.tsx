// L4: NAIVE and FLASH on the same inputs. Two engine instances; each run is its own instance, so the two never share state.
// Bars: HBM bytes, cycles, tiles. Formulas (not measurements): on-chip footprint and tile-core occupancy. Error panel:
// |engine − float attention| per element. N sweep: HBM bytes against N on seeded inputs, both schedules.
import { useMemo, useState } from "react";
import { attention } from "@/lib/reference";
import { runEngine, sweep, type RunSummary, type SweepRow } from "@/lib/engine-run";
import type { EngineModule } from "@/lib/state-schema";

interface Props {
  mod: EngineModule;
  q: number[][];
  k: number[][];
  v: number[][];
  causal: boolean;
  qrows: number;
  latency: number;
  overlap: boolean;
}

const muted = { color: "var(--muted)" } as const;
const panel = { background: "var(--panel)" } as const;
const TILE_CORE_CYCLES = 21; // one group tile: tile_start to the group's tile_done, plus the weight load share

function Bar({ label, a, b, fmtv }: { label: string; a: number; b: number; fmtv: (x: number) => string }) {
  const max = Math.max(a, b, 1);
  return (
    <div className="space-y-1">
      <div className="text-xs" style={muted}>{label}</div>
      {[{ name: "FLASH", v: a, c: "var(--accent)" }, { name: "NAIVE", v: b, c: "var(--err)" }].map((row) => (
        <div key={row.name} className="flex items-center gap-2 text-xs tabular-nums">
          <span className="w-12" style={muted}>{row.name}</span>
          <div className="h-3 flex-1 rounded" style={{ background: "var(--line)" }}>
            <div className="h-3 rounded" style={{ width: `${(row.v / max) * 100}%`, background: row.c }} />
          </div>
          <span className="w-24 text-right">{fmtv(row.v)}</span>
        </div>
      ))}
    </div>
  );
}

function ErrorGrid({ title, err, scale }: { title: string; err: number[][]; scale: number }) {
  return (
    <div>
      <div className="mb-1 text-xs" style={muted}>{title}</div>
      <div className="inline-grid gap-px" style={{ gridTemplateColumns: "repeat(4, 2.5rem)" }}>
        {err.flatMap((row, i) =>
          row.map((e, c) => (
            <div key={`${i}-${c}`} title={e.toFixed(4)} className="flex h-6 items-center justify-center text-[10px] tabular-nums"
              style={{ background: `rgba(248, 81, 73, ${Math.min(1, e / scale)})` }}>
              {e.toFixed(3)}
            </div>
          )),
        )}
      </div>
    </div>
  );
}

function SweepChart({ rows }: { rows: SweepRow[] }) {
  const W = 360;
  const H = 150;
  const maxB = Math.max(...rows.flatMap((r) => [r.flashBytes, r.naiveBytes]), 1);
  const x = (n: number) => 10 + ((n - 4) / 28) * (W - 20);
  const y = (b: number) => H - 14 - (b / maxB) * (H - 28);
  const line = (key: "flashBytes" | "naiveBytes") => rows.map((r) => `${x(r.n)},${y(r[key])}`).join(" ");
  return (
    <div className="space-y-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md" role="img" aria-label="HBM bytes against N">
        <polyline points={line("naiveBytes")} fill="none" stroke="var(--err)" strokeWidth={2} />
        <polyline points={line("flashBytes")} fill="none" stroke="var(--accent)" strokeWidth={2} />
        {rows.map((r) => (
          <text key={r.n} x={x(r.n)} y={H - 2} fontSize={9} textAnchor="middle" fill="var(--muted)">{r.n}</text>
        ))}
      </svg>
      <div className="flex gap-4 text-xs" style={muted}>
        <span><span style={{ color: "var(--accent)" }}>━</span> FLASH HBM bytes</span>
        <span><span style={{ color: "var(--err)" }}>━</span> NAIVE HBM bytes</span>
        <span>x-axis: N</span>
      </div>
      <div className="overflow-x-auto">
        <table className="text-xs tabular-nums">
          <thead>
            <tr style={muted} className="text-left">
              <th className="pr-3">N</th><th className="pr-3">FLASH bytes</th><th className="pr-3">NAIVE bytes</th>
              <th className="pr-3">FLASH cycles</th><th className="pr-3">NAIVE cycles</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.n}>
                <td className="pr-3">{r.n}</td><td className="pr-3">{r.flashBytes}</td><td className="pr-3">{r.naiveBytes}</td>
                <td className="pr-3">{r.flashCycles}</td><td className="pr-3">{r.naiveCycles}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function CompareView({ mod, q, k, v, causal, qrows, latency, overlap }: Props) {
  const [both, setBoth] = useState<{ flash: RunSummary; naive: RunSummary } | null>(null);
  const [sweepRows, setSweepRows] = useState<SweepRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const n = q.length;
  const flashQrows = Math.min(qrows, n);

  const reference = useMemo(() => attention(q, k, v, causal).out, [q, k, v, causal]);

  const runBoth = () => {
    setError(null);
    try {
      const base = { n, causal, latency };
      const flash = runEngine(mod, q, k, v, { ...base, qrows: flashQrows, overlap, naive: false });
      const naive = runEngine(mod, q, k, v, { ...base, qrows: 4, overlap: false, naive: true });
      setBoth({ flash, naive });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const runSweep = () => {
    setError(null);
    try {
      setSweepRows(sweep(mod, qrows, causal, latency, 1234));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const errs = (out: number[][]) => out.map((row, i) => row.map((x, c) => Math.abs(x - (reference[i]?.[c] ?? 0))));
  const flashErr = both ? errs(both.flash.out) : null;
  const naiveErr = both ? errs(both.naive.out) : null;
  const flatMean = (e: number[][]) => e.flat().reduce((a, b) => a + b, 0) / e.flat().length;
  const flatMax = (e: number[][]) => Math.max(...e.flat());
  const heatScale = Math.max(0.02, flashErr && naiveErr ? Math.max(flatMax(flashErr), flatMax(naiveErr)) : 0.02);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2 text-sm">
        <button className="rounded px-3 py-1" style={panel} onClick={runBoth}>Run both on these inputs</button>
        <button className="rounded px-3 py-1" style={panel} onClick={runSweep}>Sweep N = 4 to 32</button>
        <span className="self-center text-xs" style={muted}>
          FLASH: Q block {flashQrows} rows{overlap ? ", overlap on" : ""}. NAIVE: 4-row blocks. Causal {causal ? "on" : "off"}.
        </span>
      </div>
      {error && <p className="text-sm" style={{ color: "var(--err)" }}>{error}</p>}

      {both && (
        <section className="space-y-4">
          <h3 className="text-sm font-semibold">Same inputs, both schedules</h3>
          <div className="grid gap-4 md:grid-cols-2">
            <Bar label="HBM bytes (read + write)" a={both.flash.hbmRd + both.flash.hbmWr} b={both.naive.hbmRd + both.naive.hbmWr}
              fmtv={(x) => `${x} B`} />
            <Bar label="Busy cycles" a={both.flash.cycles} b={both.naive.cycles} fmtv={(x) => `${x}`} />
            <Bar label="Tiles computed" a={both.flash.tiles} b={both.naive.tiles} fmtv={(x) => `${x}`} />
            <Bar label="K/V tile loads" a={both.flash.kvLoads} b={both.naive.kvLoads} fmtv={(x) => `${x}`} />
          </div>
          <div className="overflow-x-auto">
            <table className="text-xs tabular-nums">
              <thead>
                <tr style={muted} className="text-left">
                  <th className="pr-3">Measure</th><th className="pr-3">FLASH</th><th className="pr-3">NAIVE</th><th className="pr-3">Basis</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="pr-3">On-chip footprint</td>
                  <td className="pr-3">{flashQrows * 4 + 64 + flashQrows * 13} B</td><td className="pr-3">32 B (one Z tile)</td>
                  <td>formula: Q block + K/V double buffer + O accumulators (FLASH); one 4×4 Z tile (NAIVE)</td></tr>
                <tr><td className="pr-3">Full S matrix</td><td className="pr-3">never stored</td><td className="pr-3">{n * n * 2} B per tensor, in HBM</td>
                  <td>formula: N² × 2 B for Z (NAIVE writes and reads it; P is the same size)</td></tr>
                <tr><td className="pr-3">Tile-core occupancy</td>
                  <td className="pr-3">{Math.round((both.flash.tiles * TILE_CORE_CYCLES / both.flash.cycles) * 100)}%</td><td className="pr-3">n/a</td>
                  <td>estimate: tiles × {TILE_CORE_CYCLES} / busy cycles; NAIVE per-pass cost is not counted separately</td></tr>
              </tbody>
            </table>
          </div>

          <div>
            <h3 className="mb-2 text-sm font-semibold">Error against float attention (same inputs, unquantized float reference)</h3>
            <div className="grid gap-4 md:grid-cols-2">
              <ErrorGrid title={`FLASH, max ${flatMax(flashErr ?? [[0]]).toFixed(4)}, mean ${flatMean(flashErr ?? [[0]]).toFixed(4)}`} err={flashErr ?? []} scale={heatScale} />
              <ErrorGrid title={`NAIVE, max ${flatMax(naiveErr ?? [[0]]).toFixed(4)}, mean ${flatMean(naiveErr ?? [[0]]).toFixed(4)}`} err={naiveErr ?? []} scale={heatScale} />
            </div>
            <p className="mt-2 text-xs" style={muted}>
              Both errors are dominated by the INT8 quantization of Q, K and V. NAIVE quantizes P against the global max; FLASH
              rescales as it goes. The grids share one colour scale.
            </p>
          </div>
        </section>
      )}

      {sweepRows && (
        <section className="space-y-3">
          <h3 className="text-sm font-semibold">HBM bytes against N (seeded inputs, same settings)</h3>
          <SweepChart rows={sweepRows} />
        </section>
      )}
    </div>
  );
}
