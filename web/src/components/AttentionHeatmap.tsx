// Attention probabilities P[i][j] (row i attends to column j). "engine" values are read from HBM after a NAIVE run;
// "float" values are the float reference, because FLASH never stores P.
interface Props {
  probs: number[][];
  labels: string[];
  source: "engine" | "float";
}

export default function AttentionHeatmap({ probs, labels, source }: Props) {
  const n = probs.length;
  const caption =
    source === "engine"
      ? "Attention probabilities P, read back from HBM (NAIVE)"
      : "Float reference attention (FLASH never stores P, so this is the float model)";
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold">{caption}</h3>
      <div className="overflow-x-auto">
        <div className="inline-grid gap-px" style={{ gridTemplateColumns: `repeat(${n}, 1rem)` }}>
          {probs.flatMap((row, i) =>
            row.map((p, j) => (
              <div
                key={`${i}-${j}`}
                title={`${labels[i] ?? i} → ${labels[j] ?? j}: ${p.toFixed(3)}`}
                style={{ width: "1rem", height: "1rem", background: `rgba(88, 166, 255, ${Math.min(1, p * 1.6)})` }}
              />
            )),
          )}
        </div>
      </div>
      {labels.length === n && (
        <div className="flex flex-wrap gap-x-3 text-[10px]" style={{ color: "var(--muted)" }}>
          {labels.map((l, i) => <span key={i}>{i}:{l}</span>)}
        </div>
      )}
    </div>
  );
}
