// Phase 0 only: loads the spike module built by wasm/spike/build.sh.
// Replaced by wasm-loader.ts in Phase 6.

export interface RawCounter {
  reset(): void;
  step(): void;
  getCount(): number;
  delete(): void;
}

export interface SpikeModule {
  CounterSim: new () => RawCounter;
}

export async function loadSpike(): Promise<SpikeModule> {
  if (typeof window === "undefined") {
    throw new Error("loadSpike must only be called in the browser, not during SSR");
  }
  // Built at runtime so Vite's import analysis never sees a literal public/ path.
  const url = (import.meta.env.BASE_URL as string) + "spike/counter.mjs";
  const { default: factory } = (await import(/* @vite-ignore */ url)) as {
    default: () => Promise<SpikeModule>;
  };
  return factory();
}
