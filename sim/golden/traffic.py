"""Closed-form HBM traffic per schedule. fixed.py counts bytes by walking the schedule; the test
asserts both agree, and the RTL counters must equal these numbers exactly (Phase 4/5).

Tile = 4x4. Element sizes: Q/K/V int8 (1 B), Z and O int16 (2 B), P 1 B if pf <= 7 else 2 B.
`T` = tile pairs computed: nb*nb, or nb*(nb+1)/2 when causal (tiles above the diagonal never computed).
FLASH loads each K/V tile once per Q block of `qblock` rows, so a bigger on-chip Q block means less traffic;
NAIVE's Z/P round trips do not depend on it.
"""

from fixed import BC, BR, D


def tile_pairs(n: int, causal: bool) -> int:
    nb = n // BR
    return nb * (nb + 1) // 2 if causal else nb * nb


def kv_tile_loads(n: int, causal: bool, qblock: int) -> int:
    """K (= V) tiles loaded by FLASH: once per Q block per column tile that any row in the block needs."""
    nb, gpb = n // BR, qblock // BR
    loads = 0
    for b0 in range(0, nb, gpb):
        last = min(b0 + gpb, nb) - 1
        loads += last + 1 if causal else nb
    return loads


def traffic(n: int, causal: bool, mode: str, pf: int = 8, qblock: int = BR) -> dict:
    """`qblock` (rows resident on chip per Q block) only affects FLASH."""
    nb, t = n // BR, tile_pairs(n, causal)
    q = BR * D  # one Q tile, bytes
    kv = BC * D  # one K or V tile, bytes
    z = BR * BC * 2  # one Z tile
    p = BR * BC * (1 if pf <= 7 else 2)  # one P tile
    o = BR * D * 2  # one O tile
    if mode == "FLASH":
        rd, wr = nb * q + kv_tile_loads(n, causal, qblock) * 2 * kv, nb * o
    elif mode == "NAIVE":
        rd = nb * q + t * kv  # pass 1: Q per row block, K per tile
        rd += t * z  # pass 2: Z -> row max
        rd += t * z  # pass 3: Z -> exp2
        rd += t * (p + kv)  # pass 4: P and V
        wr = t * z + t * p + nb * o  # Z, P, O
    else:
        raise ValueError(mode)
    return dict(rd=rd, wr=wr, total=rd + wr, tiles=t, skipped=nb * nb - t)
