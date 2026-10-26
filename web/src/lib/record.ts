// Records a whole FLASH run tile by tile (the row state at each tile_done pulse), for the docs demos. Same observation
// rule as the visualizer: the tile completed on an edge is order[tiles - 1] of that run.
import { quantize } from "@/lib/quantize";
import { flashTileOrder } from "@/lib/schedule";
import type { EngineModule } from "@/lib/state-schema";
import type { TileSample } from "@/lib/softmax-cell";

const MAX_CYCLES = 400_000;

export function recordFlashRun(
  mod: EngineModule,
  q: number[][],
  k: number[][],
  v: number[][],
  causal: boolean,
  qrows: number,
  latency: number,
): TileSample[] {
  const n = q.length;
  const qz = quantize(q, k, v);
  const order = flashTileOrder(n, qrows, causal);
  const sim = new mod.TinyAttnSim();
  const samples: TileSample[] = [];
  const observe = (s: { tileDone: boolean; tiles: number }) => {
    if (!s.tileDone) return;
    const ref = order[s.tiles - 1];
    if (!ref) return;
    const t = sim.getTile();
    samples.push({ qi: ref.qi, kj: ref.kj, z: t.z, m: t.m, l: t.l, alpha: t.alpha });
  };
  try {
    sim.configure(n, qrows, causal, 0, false, latency, qz.cfgScale);
    sim.loadQKV(qz.q, qz.k, qz.v);
    sim.start();
    for (let i = 0; i < MAX_CYCLES; i++) {
      const s = sim.step();
      observe(s);
      if (s.done) {
        observe(sim.step());
        break;
      }
    }
  } finally {
    sim.delete();
  }
  return samples;
}
