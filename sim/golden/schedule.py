"""Cycle model of rtl/flash_ctrl.sv (Phase 4, no K/V overlap).

Counts the cycles the engine is busy: from the first state after `start` through C_DONE (inclusive), which is what the
`cycles` counter in rtl/perf_counters.sv measures. Each term follows one state of the controller:

  C_BLK                      1
  C_LDQ + C_LDQ_W            1 + (L + 4*bg + 2)     DMA read of 4*bg words, L = cfg_hbm_latency
  per K/V tile kj:
    C_LDK + C_LDK_W          1 + (L + 6)            DMA read of 4 words (K)
    C_LDV + C_LDV_W          1 + (L + 6)            DMA read of 4 words (V)
    C_WLOAD                  4
    per row group gi of the block:
      masked (causal)        1                      C_TILE, skipped
      tiled                  1 + 20                 C_TILE (tile_start) + C_TILE_W (tile_done at 20)
    C_TILE at end            1
  C_NORM + C_NORM_W          1 + (4*bg + 1)         normalize bg*4 rows
  C_STO + C_STO_W            1 + (8*bg + 1)         posted DMA write of 8*bg words
  C_DONE                     1

Derivation of the DMA terms (see test_dma.py): a read burst of W words takes L + W + 2 cycles from its start cycle to
the cycle its done pulse is seen; a write burst takes W + 1.
"""

TILE_BUSY = 21  # C_TILE (tile_start) + C_TILE_W for the 20-cycle tile pipeline


def cycles(n: int, causal: bool, qrows: int, latency: int) -> int:
    nb, g_per_blk = n // 4, qrows // 4
    total = 1  # C_DONE
    gb = 0
    while gb < nb:
        bg = min(g_per_blk, nb - gb)
        total += 1                          # C_BLK
        total += 1 + (latency + 4 * bg + 2)  # Q load
        last = gb + bg - 1 if causal else nb - 1
        for kj in range(last + 1):
            total += 2 * (1 + latency + 6)  # K and V loads
            total += 4                      # weight load
            for gi in range(bg):
                g = gb + gi
                total += 1 if (causal and kj > g) else TILE_BUSY
            total += 1                      # end of this K/V tile
        total += 1 + (4 * bg + 1)           # normalize
        total += 1 + (8 * bg + 1)           # store O
        gb += g_per_blk
    return total
