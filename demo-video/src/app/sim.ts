// Everything the replica UI shows comes from src/trace.json, recorded from the real WASM engine (scripts/trace.mts).
import trace from "../trace.json";

export const FLASH_STATES = ["IDLE","BLK","LDQ","LDQ_W","LDK","LDK_W","LDV","LDV_W","WLOAD","TILE","TILE_W","WAITPF","NORM","NORM_W","STO","STO_W","DONE"] as const;
export const T = trace as any;
export const LABELS: string[] = T.toy.labels;
export const RUN = T.toy.run;
export const TOTAL_CLOCKS: number = RUN.steps;
RUN.clocks.push({ kv: RUN.kvLoads, c: RUN.steps, st: 16, t: RUN.tiles, rd: RUN.rd, wr: RUN.wr });
export const EXPECTED = { tiles: RUN.tiles, rd: RUN.rd, wr: RUN.wr };

// FLASH tile order for N=12, Q block 4 rows, causal (web/src/lib/schedule.ts)
export const ORDER: { qi: number; kj: number }[] = (() => {
  const nb = 3, o: { qi: number; kj: number }[] = [];
  for (let gb = 0; gb < nb; gb++) for (let kj = 0; kj <= gb; kj++) o.push({ qi: gb, kj });
  return o;
})();

export interface Snap { clock: number; state: string; tiles: number; kv: number; rd: number; wr: number; done: number; inFlight: number; last: any }
export function snapAt(clock: number): Snap {
  const cs: any[] = RUN.clocks;
  let s = cs[0];
  for (const c of cs) { if (c.c <= clock) s = c; else break; }
  const tt: any[] = RUN.tileTrace;
  const done = tt.filter((x) => x.clk <= clock).length;
  return {
    clock: Math.round(clock), state: FLASH_STATES[s.st] ?? "IDLE", tiles: s.t, kv: s.kv, rd: s.rd, wr: s.wr, done,
    inFlight: s.t > done ? done : -1, last: tt[Math.max(0, done - 1)],
  };
}
export const SWEEP: { n: number; flashBytes: number; naiveBytes: number; flashCycles: number; naiveCycles: number }[] = T.sweep;
export const CMP = T.cmp;
