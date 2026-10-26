// The online softmax on the toy sentence: the engine records each tile's row state; SoftmaxCellView shows it.
import { useEffect, useState } from "react";
import SoftmaxCellView from "@/components/SoftmaxCellView";
import { loadEngine } from "@/lib/engine";
import { PRESETS } from "@/lib/presets";
import { recordFlashRun } from "@/lib/record";
import type { TileSample } from "@/lib/softmax-cell";
import type { EngineModule } from "@/lib/state-schema";

const preset = PRESETS[0]!;

export default function OnlineSoftmaxDemo() {
  const [mod, setMod] = useState<EngineModule | null>(null);
  const [samples, setSamples] = useState<TileSample[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadEngine().then(setMod).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    if (!mod) return;
    try {
      setSamples(recordFlashRun(mod, preset.q, preset.k, preset.v, preset.causal, 4, 10));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [mod]);

  if (error) return <p style={{ color: "var(--err)" }}>{error}</p>;
  if (!samples) return <p className="text-sm" style={{ color: "var(--muted)" }}>Running the engine on the toy sentence…</p>;
  return (
    <div className="demo">
      <p className="mb-3 text-sm" style={{ color: "var(--muted)" }}>
        The toy sentence on the engine, FLASH schedule. Pick a token to see the running state its row goes through.
      </p>
      <SoftmaxCellView samples={samples} q={preset.q} k={preset.k} causal={preset.causal} labels={preset.labels} naive={false} />
    </div>
  );
}
