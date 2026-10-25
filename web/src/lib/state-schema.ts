// Mirrors wasm/harness/harness.h and docs/STATE_SCHEMA.md. Change both together.

/** Per-clock summary returned by step(), snapshot(). */
export interface Snapshot {
  cycle: number;
  ctrlState: number;
  busy: boolean;
  done: boolean;
  hbmRdBytes: number;
  hbmWrBytes: number;
  cycles: number;
  tiles: number;
  kvLoads: number;
}

/** Tile-core state of the most recent tile (4 rows of one row group). */
export interface TileState {
  curGrp: number;
  srow: number;
  s: number[]; // 16 values, row-major (4 rows x 4 key columns)
  z: number[];
  p: number[];
  alpha: number[]; // 4 rows
  m: number[];
  l: number[];
  o: number[]; // 16 values, row-major (4 rows x 4 columns)
}

export interface EngineInstance {
  reset(): void;
  configure(n: number, qrows: number, causal: boolean, mode: number, overlap: boolean, latency: number, scale: number): void;
  loadQKV(q: number[], k: number[], v: number[]): void;
  start(): void;
  step(): Snapshot;
  runToDone(maxCycles: number): number;
  readOutput(): number[];
  readHbm(addr: number, len: number): number[];
  readProbabilities(): number[];
  snapshot(): Snapshot;
  isDone(): boolean;
  getTile(): TileState;
  delete(): void;
}

export interface EngineModule {
  TinyAttnSim: new () => EngineInstance;
}

/** Controller state names (flash_ctrl.sv and naive_ctrl.sv, the enum order). */
export const FLASH_STATES = [
  "IDLE", "BLK", "LDQ", "LDQ_W", "LDK", "LDK_W", "LDV", "LDV_W", "WLOAD",
  "TILE", "TILE_W", "WAITPF", "NORM", "NORM_W", "STO", "STO_W", "DONE",
] as const;

export const NAIVE_STATES = [
  "IDLE", "BLK", "LDQ", "LDQ_W", "ISSUE", "WAIT", "WLOAD", "TILE", "TILE_W",
  "STORE", "STORE_W", "NEXT", "NORM", "NORM_W", "STO", "STO_W", "DONE",
] as const;
