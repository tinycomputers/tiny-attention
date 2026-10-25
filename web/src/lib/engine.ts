// Loads the engine module. Client only: called from a client:only island, never during SSR.
import type { EngineModule } from "@/lib/state-schema";

let modulePromise: Promise<EngineModule> | null = null;

export function loadEngine(): Promise<EngineModule> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("loadEngine must only run in the browser"));
  }
  if (!modulePromise) {
    // Built at runtime so Vite's import analysis never sees a literal public/ path (see astro.config.mjs).
    const url = (import.meta.env.BASE_URL as string) + "tiny_attn.mjs";
    modulePromise = (import(/* @vite-ignore */ url) as Promise<{ default: () => Promise<EngineModule> }>)
      .then(({ default: factory }) => factory())
      .catch((e: unknown) => {
        modulePromise = null;
        throw new Error(`engine not found: run bash wasm/build.sh (${e instanceof Error ? e.message : String(e)})`);
      });
  }
  return modulePromise;
}
