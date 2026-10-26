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
  tileDone: boolean; // one-cycle pulse: a tile completed on the last edge
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

/** Datapath wires of the two systolic arrays. Index r*4 + c is PE (r, c). */
export interface Datapath {
  mm1W: number[]; // Kᵀ weights, int8
  mm1A: number[]; // Q activations, int8
  mm1Psum: number[]; // S partial sums, int17
  mm2W: number[]; // V weights, int8
  mm2A: number[]; // P activations, uint9
  mm2Psum: number[]; // P·V partial sums, int18
}

/** K and V tile buffers. k[(buf*4 + row)*4 + col] is K[row][col] of buffer buf (int8). sel: buffer in use. */
export interface KVBuffers {
  sel: number;
  k: number[]; // 32
  v: number[]; // 32
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
  getKVBuffers(): KVBuffers;
  getDatapath(): Datapath;
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
