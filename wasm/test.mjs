// Headless test of the TinyAttn WebAssembly module against the golden fixtures (wasm/fixtures, exported by
// sim/golden/export_fixtures.py). Two instances run side by side with their steps interleaved: any state shared
// between instances would change an output or a cycle count. Per run it checks:
//   O (every value) == golden, HBM bytes == traffic.py, tiles and K/V loads == traffic.py, busy cycles == fixture.
// Usage: node wasm/test.mjs   (after bash wasm/build.sh)
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// ENGINE_DIR selects which build to test (CI tests the committed files before rebuilding them); default web/public.
const engineDir = process.env.ENGINE_DIR ?? join(root, "web/public");
const { default: createTinyAttn } = await import(join(engineDir, "tiny_attn.mjs"));
const mod = await createTinyAttn();

// engine runs only; quantize_cases.json holds quantization references for the web tests
const fixtures = readdirSync(join(root, "wasm/fixtures"))
  .filter((f) => f.endsWith(".json") && !f.startsWith("quantize_"))
  .sort()
  .map((f) => JSON.parse(readFileSync(join(root, "wasm/fixtures", f), "utf8")));

let failures = 0;
const check = (ok, msg) => {
  if (!ok) {
    failures++;
    console.log(`  FAIL ${msg}`);
  }
};

// FLASH tile order (web/src/lib/schedule.ts): Q blocks outer, K/V tiles, row groups; causal skips masked tiles.
function flashOrder(n, qrows, causal) {
  const nb = n / 4;
  const g = qrows / 4;
  const order = [];
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

function load(fx) {
  const sim = new mod.TinyAttnSim();
  sim.configure(fx.n, fx.qrows, fx.causal, fx.mode === "NAIVE" ? 1 : 0, fx.overlap, fx.latency, fx.cfgScale);
  sim.loadQKV(fx.q, fx.k, fx.v);
  sim.start();
  return sim;
}

// Run all fixtures together, one clock at a time, round-robin over the instances.
const sims = fixtures.map((fx) => ({ fx, sim: load(fx), last: null, finished: false }));
const maxCycles = 400_000;
for (let t = 0; t < maxCycles && sims.some((s) => !s.finished); t++) {
  for (const s of sims) {
    if (s.finished) continue;
    s.last = s.sim.step();
    if (s.last.tileDone && s.fx.mode === "FLASH") {
      // one completion per tile; the running row max m never decreases within a row group
      s.tileDones = (s.tileDones ?? 0) + 1;
      const t = s.sim.getTile();
      s.order = s.order ?? flashOrder(s.fx.n, s.fx.qrows, s.fx.causal);
      const key = s.order[s.tileDones - 1].qi;  // global row group of this completion
      s.prevM = s.prevM ?? new Map();
      const prev = s.prevM.get(key);
      if (prev) {
        for (let r = 0; r < 4; r++) {
          if (t.m[r] < prev[r]) {
            check(false, `${s.fx.name}: m decreased for group ${key} row ${r}: ${prev[r]} -> ${t.m[r]}`);
          }
        }
      }
      s.prevM.set(key, t.m.slice());
    }
    if (s.last.done) {
      // the DONE cycle is counted by the busy-cycle counter at its end: one more edge makes the counters final
      s.last = s.sim.step();
      s.finished = true;
    }
  }
}

for (const { fx, sim, last, finished, tileDones } of sims) {
  console.log(`${fx.name} (${fx.mode} N=${fx.n} causal=${fx.causal} qrows=${fx.qrows} overlap=${fx.overlap} L=${fx.latency})`);
  check(finished, "engine did not finish");
  const out = Array.from(sim.readOutput());
  const bad = out.findIndex((v, i) => v !== fx.out[i]);
  check(bad < 0, `O differs at ${bad}: expected ${fx.out[bad]}, got ${out[bad]}`);
  check(last.hbmRdBytes === fx.hbmRdBytes, `read bytes ${last.hbmRdBytes} != ${fx.hbmRdBytes}`);
  check(last.hbmWrBytes === fx.hbmWrBytes, `write bytes ${last.hbmWrBytes} != ${fx.hbmWrBytes}`);
  check(last.tiles === fx.tiles, `tiles ${last.tiles} != ${fx.tiles}`);
  if (fx.mode === "FLASH") check(tileDones === fx.tiles, `tile completions ${tileDones} != ${fx.tiles}`);
  check(last.kvLoads === fx.kvLoads, `K/V loads ${last.kvLoads} != ${fx.kvLoads}`);
  check(last.cycles === fx.cycles, `busy cycles ${last.cycles} != ${fx.cycles}`);
  if (fx.lastTile) {
    // tile state of the last computed tile, against the golden trace entry for that tile
    const t = sim.getTile();
    const G = fx.qrows / 4;
    const lt = fx.lastTile;
    check(t.curGrp === lt.qi - Math.floor(lt.qi / G) * G, `tile group ${t.curGrp}`);
    check(JSON.stringify(t.m) === JSON.stringify(lt.m), `tile m ${t.m} != ${lt.m}`);
    check(JSON.stringify(t.l) === JSON.stringify(lt.l), `tile l ${t.l} != ${lt.l}`);
    check(JSON.stringify(t.alpha) === JSON.stringify(lt.alpha), `tile alpha ${t.alpha} != ${lt.alpha}`);
    check(JSON.stringify(t.o) === JSON.stringify(lt.o), "tile O differs from golden");
    console.log(`  last tile (qi=${lt.qi}, kj=${lt.kj}) state checked`);
  }
  {
    // The K/V buffer in use holds the last K/V tile loaded, which is K/V tile n/4-1 for every fixture (the last
    // block's last tile). Its rows are the last four rows of the quantized K and V.
    const kv = sim.getKVBuffers();
    const lastK = fx.k.slice((fx.n - 4) * 4, fx.n * 4);
    const lastV = fx.v.slice((fx.n - 4) * 4, fx.n * 4);
    const bufK = kv.k.slice(kv.sel * 16, kv.sel * 16 + 16);
    const bufV = kv.v.slice(kv.sel * 16, kv.sel * 16 + 16);
    check(JSON.stringify(bufK) === JSON.stringify(lastK), `K buffer ${kv.sel} != last K tile`);
    check(JSON.stringify(bufV) === JSON.stringify(lastV), `V buffer ${kv.sel} != last V tile`);
    console.log(`  K/V buffer ${kv.sel} holds the last K/V tile`);
  }
  if (fx.probs) {
    const probs = Array.from(sim.readProbabilities());
    const badP = probs.findIndex((v, i) => v !== fx.probs[i]);
    check(badP < 0, `probability ${badP} differs: expected ${fx.probs[badP]}, got ${probs[badP]}`);
    console.log(`  probabilities (${fx.n}x${fx.n}) checked`);
  }
  console.log(`  cycles ${last.cycles}  read ${last.hbmRdBytes} B  write ${last.hbmWrBytes} B  tiles ${last.tiles}`);
  sim.delete();
}

// A fresh instance must reproduce the same run (no state left over from earlier instances).
{
  const fx = fixtures[0];
  const sim = load(fx);
  let snap;
  for (let t = 0; t < maxCycles; t++) {
    snap = sim.step();
    if (snap.done) break;
  }
  snap = sim.step();
  check(Array.from(sim.readOutput()).every((v, i) => v === fx.out[i]), "fresh instance output differs");
  check(snap.cycles === fx.cycles, "fresh instance cycles differ");
  sim.delete();
  console.log(`fresh instance (${fx.name}): checked`);
}

console.log(failures === 0 ? "PASS" : `FAILED (${failures} checks)`);
process.exit(failures === 0 ? 0 : 1);
