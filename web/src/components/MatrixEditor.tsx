// Editable Q, K or V matrix (one row per token, four columns). Values are the float inputs the browser quantizes.
interface Props {
  title: string;
  values: number[][];
  onChange: (row: number, col: number, value: number) => void;
}

export default function MatrixEditor({ title, values, onChange }: Props) {
  const cols = values[0]?.length ?? 4;
  return (
    <div>
      <div className="mb-1 text-xs" style={{ color: "var(--muted)" }}>{title}</div>
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {values.flatMap((row, r) =>
          row.map((value, c) => (
            <input
              key={`${r}-${c}`}
              type="number"
              step="0.1"
              value={value}
              aria-label={`${title} token ${r} dimension ${c}`}
              className="w-full rounded px-1 py-0.5 font-mono text-xs"
              style={{ background: "var(--panel)" }}
              onChange={(e) => {
                const x = e.target.valueAsNumber;
                if (Number.isFinite(x)) onChange(r, c, x);
              }}
            />
          )),
        )}
      </div>
    </div>
  );
}
