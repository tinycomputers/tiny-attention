import { useEffect, useState } from "react";
import { loadSpike } from "@/lib/spike-loader";

type State =
  | { status: "loading" }
  | { status: "running"; a: number; b: number }
  | { status: "error"; message: string };

// Two instances of ONE Verilated model, ticking at different speeds. L4 (Naive vs Flash) relies on
// instances never sharing state: stepping A must not move B.
const A_MS = 150;
const B_MS = 600;

export default function SpikeCounter() {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    const cleanups: Array<() => void> = [];

    loadSpike()
      .then((mod) => {
        if (cancelled) return;
        const a = new mod.CounterSim();
        const b = new mod.CounterSim();
        a.reset();
        b.reset();
        setState({ status: "running", a: a.getCount(), b: b.getCount() });
        const ta = setInterval(() => {
          a.step();
          setState((s) => (s.status === "running" ? { ...s, a: a.getCount() } : s));
        }, A_MS);
        const tb = setInterval(() => {
          b.step();
          setState((s) => (s.status === "running" ? { ...s, b: b.getCount() } : s));
        }, B_MS);
        cleanups.push(() => {
          clearInterval(ta);
          clearInterval(tb);
          a.delete();
          b.delete();
        });
      })
      .catch((e: unknown) =>
        setState({
          status: "error",
          message:
            (e instanceof Error ? e.message : String(e)) +
            " - run `bash wasm/spike/build.sh` first.",
        })
      );

    return () => {
      cancelled = true;
      cleanups.forEach((fn) => fn());
    };
  }, []);

  if (state.status !== "running") {
    return (
      <pre
        className="mt-6 rounded-md p-4 text-lg"
        style={{
          background: "var(--panel)",
          color: state.status === "error" ? "var(--err)" : "var(--muted)",
        }}
        data-testid="spike-output"
      >
        {state.status === "loading" ? "Loading WASM..." : `ERROR: ${state.message}`}
      </pre>
    );
  }

  return (
    <div className="mt-6 grid grid-cols-2 gap-4" data-testid="spike-output">
      {(
        [
          ["A", state.a, A_MS],
          ["B", state.b, B_MS],
        ] as const
      ).map(([name, count, ms]) => (
        <div key={name} className="rounded-md p-4" style={{ background: "var(--panel)" }}>
          <div className="text-sm" style={{ color: "var(--muted)" }}>
            Instance {name} &middot; 1 tick / {ms} ms
          </div>
          <div
            className="mt-1 text-4xl font-bold tabular-nums"
            style={{ color: "var(--ok)" }}
            data-testid={`count-${name}`}
          >
            {count}
          </div>
        </div>
      ))}
    </div>
  );
}
