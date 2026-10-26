// HBM traffic so far against the expected total for this configuration (generated from sim/golden/traffic.py).
interface Props {
  read: number;
  write: number;
  expected: { rd: number; wr: number } | null;
}

function Bar({ label, value, total, color }: { label: string; value: number; total: number; color: string }) {
  const pct = total > 0 ? Math.min(100, (value / total) * 100) : 0;
  return (
    <div>
      <div className="flex justify-between text-xs tabular-nums">
        <span style={{ color: "var(--muted)" }}>{label}</span>
        <span>{value} / {total} B</span>
      </div>
      <div className="mt-1 h-2 w-full overflow-hidden rounded" style={{ background: "var(--panel)" }}>
        <div className="h-full" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function HbmMeter({ read, write, expected }: Props) {
  if (!expected) {
    return <p className="text-sm" style={{ color: "var(--muted)" }}>No traffic reference for this configuration.</p>;
  }
  return (
    <div className="space-y-2">
      <Bar label="Read" value={read} total={expected.rd} color="var(--accent)" />
      <Bar label="Write" value={write} total={expected.wr} color="var(--ok)" />
    </div>
  );
}
