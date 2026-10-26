// The order in which the FLASH controller computes tiles (flash_ctrl.sv): Q blocks outer; within a block, K/V tiles kj
// in order; within a K/V tile, the row groups of the block in order, skipping groups above the causal diagonal.
// The tiles counter counts tile starts, so tile k (0-based) in this order is the one in flight after k+1 starts.

export interface TileRef {
  qi: number; // row group (4 rows), global
  kj: number; // K/V tile
}

export function flashTileOrder(n: number, qrows: number, causal: boolean): TileRef[] {
  const nb = n / 4;
  const g = qrows / 4;
  const order: TileRef[] = [];
  for (let gb = 0; gb < nb; gb += g) {
    const bg = Math.min(g, nb - gb);
    const last = causal ? gb + bg - 1 : nb - 1;
    for (let kj = 0; kj <= last; kj++) {
      for (let gi = 0; gi < bg; gi++) {
        const qi = gb + gi;
        if (causal && kj > qi) continue;
        order.push({ qi, kj });
      }
    }
  }
  return order;
}

/** NAIVE tile pairs, block by block (naive_ctrl.sv): every K/V tile of block qi, for each qi. */
export function naiveTileOrder(n: number, causal: boolean): TileRef[] {
  const nb = n / 4;
  const order: TileRef[] = [];
  for (let qi = 0; qi < nb; qi++) {
    const last = causal ? qi : nb - 1;
    for (let kj = 0; kj <= last; kj++) order.push({ qi, kj });
  }
  return order;
}
