// The K and V tile buffers the engine holds, read from the RTL (dbg_kbuf, dbg_vbuf). Buffer 1 is only used by the
// overlap schedule; the buffer in use is outlined.
import type { KVBuffers } from "@/lib/state-schema";

interface Props {
  kv: KVBuffers | null;
}

function Tile({ title, values }: { title: string; values: number[] }) {
  const rows: number[][] = [];
  for (let r = 0; r < 4; r++) rows.push(values.slice(r * 4, r * 4 + 4));
  return (
    <div>
      <div className="text-[10px]" style={{ color: "var(--muted)" }}>{title}</div>
      <div className="inline-grid gap-x-2 font-mono text-xs tabular-nums" style={{ gridTemplateColumns: "repeat(4, auto)" }}>
        {rows.flatMap((row, i) => row.map((v, j) => <span key={`${i}-${j}`}>{v}</span>))}
      </div>
    </div>
  );
}

export default function SramPanel({ kv }: Props) {
  if (!kv) {
    return <p className="text-sm" style={{ color: "var(--muted)" }}>The buffers fill once a run starts.</p>;
  }
  return (
    <div className="grid grid-cols-2 gap-3">
      {[0, 1].map((b) => (
        <div
          key={b}
          className="space-y-2 rounded p-2"
          style={{ background: "var(--panel)", outline: kv.sel === b ? "2px solid var(--accent)" : "none" }}
        >
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            Buffer {b}{kv.sel === b ? " · in use" : ""}
          </div>
          <Tile title="K tile" values={kv.k.slice(b * 16, b * 16 + 16)} />
          <Tile title="V tile" values={kv.v.slice(b * 16, b * 16 + 16)} />
        </div>
      ))}
    </div>
  );
}
