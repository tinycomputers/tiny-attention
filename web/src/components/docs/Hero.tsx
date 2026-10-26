// Landing hero: the FLASH schedule for N = 16 (Q block of 8 rows, causal) sweeping the tile grid. Pure React, no engine.
import { useEffect, useMemo, useState } from "react";
import { flashTileOrder } from "@/lib/schedule";

const N = 16;
const QROWS = 8;
const STEP_MS = 180;

export default function Hero() {
  const order = useMemo(() => flashTileOrder(N, QROWS, true), []);
  const [step, setStep] = useState(0);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setStep(order.length);
      return;
    }
    const id = window.setInterval(() => setStep((s) => (s + 1) % (order.length + 8)), STEP_MS);
    return () => window.clearInterval(id);
  }, [order.length]);

  const blocks = N / 4;
  const done = new Set<string>();
  order.slice(0, Math.min(step, order.length)).forEach((t) => done.add(`${t.qi},${t.kj}`));
  const current = step < order.length ? order[step] : undefined;
  const cur = current ? `${current.qi},${current.kj}` : "";

  return (
    <div className="space-y-3">
      <div className="inline-grid gap-1" style={{ gridTemplateColumns: `auto repeat(${blocks}, 2.5rem)` }}
        role="img" aria-label="FLASH tile schedule: row groups by K/V tiles, causal mask, tiles filling in order">
        <div />
        {Array.from({ length: blocks }, (_, kj) => (
          <div key={`h${kj}`} className="text-center text-[10px]" style={{ color: "var(--muted)" }}>kv{kj}</div>
        ))}
        {Array.from({ length: blocks }, (_, qi) => (
          <div key={`r${qi}`} className="contents">
            <div className="pr-2 text-right text-[10px]" style={{ color: "var(--muted)" }}>q{qi}</div>
            {Array.from({ length: blocks }, (_, kj) => {
              const masked = kj > qi;
              const key = `${qi},${kj}`;
              const bg = masked ? "transparent" : key === cur ? "var(--accent)" : done.has(key) ? "var(--ok)" : "var(--panel)";
              return (
                <div key={key} className="h-10 rounded"
                  style={{ background: bg, border: "1px solid var(--line)", opacity: masked ? 0.2 : 1 }} />
              );
            })}
          </div>
        ))}
      </div>
      <p className="text-xs" style={{ color: "var(--muted)" }}>
        Each cell is a 4×4 tile of the attention matrix. Blue is computing, green is done, faded tiles are masked and never
        computed. The full matrix is never stored.
      </p>
    </div>
  );
}
