// Tab strip for the views (L1 to L4). Same look as tiny-cpu's tabs.
export interface LevelDef<T extends string> {
  id: T;
  label: string;
}

interface Props<T extends string> {
  levels: LevelDef<T>[];
  value: T;
  onChange: (id: T) => void;
}

export default function LevelTabs<T extends string>({ levels, value, onChange }: Props<T>) {
  return (
    <div role="tablist" aria-label="Views" className="flex flex-wrap gap-1 border-b" style={{ borderColor: "var(--line)" }}>
      {levels.map((l) => {
        const selected = l.id === value;
        return (
          <button
            key={l.id}
            role="tab"
            aria-selected={selected}
            aria-controls={`panel-${l.id}`}
            id={`tab-${l.id}`}
            onClick={() => onChange(l.id)}
            className="rounded-t-md border border-b-0 px-3 py-1.5 text-sm"
            style={{
              borderColor: selected ? "var(--accent)" : "transparent",
              color: selected ? "var(--fg)" : "var(--muted)",
              background: selected ? "var(--panel)" : "transparent",
            }}
          >
            {l.label}
          </button>
        );
      })}
    </div>
  );
}
