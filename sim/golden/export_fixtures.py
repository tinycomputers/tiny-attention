"""Export golden fixtures for the WASM Node test (wasm/test.mjs).

Each fixture holds one engine run: the quantized inputs, the config, the expected O (int16, Q8.8 V-LSB), the expected
HBM bytes, tile count and K/V loads (traffic.py), and the expected busy cycles. Cycles come from the closed-form model
for sequential FLASH, and from sim/perf_baseline.json for overlap and NAIVE (those are pinned baselines).

    python sim/golden/export_fixtures.py     # writes wasm/fixtures/*.json
"""

import json
import random
from pathlib import Path

import numpy as np

import fixed
from schedule import cycles as sched_cycles
from traffic import kv_tile_loads, traffic

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "wasm" / "fixtures"
BASELINE = json.loads((ROOT / "sim" / "perf_baseline.json").read_text())

CASES = [
    dict(name="flash_seq", mode="FLASH", n=16, causal=False, qrows=8, overlap=False, latency=10, seed=7),
    dict(name="flash_overlap_causal", mode="FLASH", n=16, causal=True, qrows=4, overlap=True, latency=10, seed=8),
    dict(name="naive_causal", mode="NAIVE", n=16, causal=True, qrows=4, overlap=False, latency=10, seed=9),
]


def case_inputs(seed, n):
    rng = np.random.default_rng(seed)
    qf = rng.normal(0, 2.5, (n, 4))
    kf = rng.normal(0, 2.5, (n, 4))
    vf = rng.normal(0, 1.0, (n, 4))
    return fixed.quantize(qf, kf, vf)


def expected_cycles(c):
    if c["mode"] == "FLASH" and not c["overlap"]:
        return sched_cycles(c["n"], c["causal"], c["qrows"], c["latency"])
    key = (f"naive/N{c['n']}/c{int(c['causal'])}/L{c['latency']}" if c["mode"] == "NAIVE"
           else f"flash/N{c['n']}/c{int(c['causal'])}/q{c['qrows']}/o{int(c['overlap'])}/L{c['latency']}")
    return BASELINE[key]


def build_fixture(c):
    qz = case_inputs(c["seed"], c["n"])
    naive = c["mode"] == "NAIVE"
    qblock = 4 if naive else c["qrows"]
    res = fixed.naive(qz, c["n"], c["causal"]) if naive else fixed.flash(qz, c["n"], c["causal"], qblock=qblock)
    t = traffic(c["n"], c["causal"], c["mode"], qblock=qblock)
    return dict(
        name=c["name"],
        mode=c["mode"],
        n=c["n"],
        causal=c["causal"],
        qrows=qblock,
        overlap=c["overlap"] and not naive,
        latency=c["latency"],
        cfgScale=int(qz.cfg_scale),
        q=qz.q.reshape(-1).tolist(),
        k=qz.k.reshape(-1).tolist(),
        v=qz.v.reshape(-1).tolist(),
        out=res.out.reshape(-1).astype(int).tolist(),
        hbmRdBytes=t["rd"],
        hbmWrBytes=t["wr"],
        tiles=t["tiles"],
        kvLoads=t["tiles"] if naive else kv_tile_loads(c["n"], c["causal"], qblock),
        cycles=expected_cycles(c),
    )


def fixture_text(c):
    return json.dumps(build_fixture(c)) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for c in CASES:
        text = fixture_text(c)
        (OUT / f"{c['name']}.json").write_text(text)
        f = json.loads(text)
        print(f"{c['name']}: cycles {f['cycles']} bytes rd {f['hbmRdBytes']} wr {f['hbmWrBytes']} tiles {f['tiles']}")


if __name__ == "__main__":
    main()
