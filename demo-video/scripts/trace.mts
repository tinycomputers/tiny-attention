// Runs the REAL TinyAttention WASM engine and records what the video shows. Nothing here is invented.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { quantize } from "../../web/src/lib/quantize.ts";
import { PRESETS } from "../../web/src/lib/presets.ts";
function seededInputs(n: number, seed: number) {
  let s = seed >>> 0;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  const mat = (amp: number) => Array.from({ length: n }, () => [0, 1, 2, 3].map(() => (next() * 2 - 1) * amp));
  return { q: mat(2), k: mat(2), v: mat(1) };
}

const here = dirname(fileURLToPath(import.meta.url));
const { default: create } = await import(join(here, "../../web/public/tiny_attn.mjs"));
const mod = await create();

function run(q: number[][], k: number[][], v: number[][], c: any, sample = false) {
  const qz = quantize(q, k, v);
  const sim = new mod.TinyAttnSim();
  sim.configure(c.n, c.qrows, c.causal, c.naive ? 1 : 0, c.overlap, c.latency, qz.cfgScale);
  sim.loadQKV(qz.q, qz.k, qz.v);
  sim.start();
  const tiles: any[] = [];
  const clocks: any[] = [];
  let last: any;
  let dpLast: any = null;
  let dpScore = -1;
  let steps = 0;
  for (let t = 0; t < 400000; t++) {
    last = sim.step();
    if (sample && t % 4 === 0) clocks.push({ kv: last.kvLoads, c: t, st: last.ctrlState, t: last.tiles, rd: last.hbmRdBytes, wr: last.hbmWrBytes });
    if (sample && !c.naive) {
      const d = sim.getDatapath();
      const nz = (x: any) => [...x].filter((v: number) => v !== 0).length;
      const s1 = nz(d.mm1A) + nz(d.mm1Psum), s2 = nz(d.mm2A) + nz(d.mm2Psum);
      dpLast = dpLast ?? {};
      if (s1 >= (dpLast.s1 ?? -1)) Object.assign(dpLast, { s1, w1: [...d.mm1W], a1: [...d.mm1A], p1: [...d.mm1Psum] });
      if (s2 >= (dpLast.s2 ?? -1)) Object.assign(dpLast, { s2, w2: [...d.mm2W], a2: [...d.mm2A], p2: [...d.mm2Psum] });
    }
    if (sample && last.tileDone && !c.naive) {
      const s = sim.getTile();
      const dpOut = dpLast; dpLast = null;
      tiles.push({ dp: dpOut, s: [...s.s], kv: (() => { const b = sim.getKVBuffers(); return { sel: b.sel, k: [...b.k], v: [...b.v] }; })(), clk: t, grp: s.curGrp, m: [...s.m], l: [...s.l], alpha: [...s.alpha], p: [...s.p], z: [...s.z], o: [...s.o] });
    }
    if (last.done) { steps = t + 1; break; }
  }
  last = sim.step();
  const res = { steps, cycles: last.cycles, tiles: last.tiles, kvLoads: last.kvLoads, rd: last.hbmRdBytes, wr: last.hbmWrBytes, tileTrace: tiles, clocks };
  sim.delete();
  return res;
}

const toy = PRESETS[0];
const cfgToy = { n: 12, qrows: 4, causal: true, naive: false, overlap: false, latency: 10 };
const toyRun = run(toy.q, toy.k, toy.v, cfgToy, true);

const { q, k, v } = seededInputs(32, 7);
const base = { n: 32, qrows: 8, causal: false, overlap: false, latency: 10 };
const flash = run(q, k, v, { ...base, naive: false });
const naive = run(q, k, v, { ...base, qrows: 4, naive: true });
const flashOv = run(q, k, v, { ...base, naive: false, overlap: true });

const sweepRows: any[] = [];
for (let n = 4; n <= 32; n += 4) {
  const x = seededInputs(n, 1234 + n);
  const b = { n, qrows: Math.min(4, n), causal: false, overlap: false, latency: 10 };
  const f = run(x.q, x.k, x.v, { ...b, naive: false }); const nv = run(x.q, x.k, x.v, { ...b, naive: true });
  sweepRows.push({ n, flashBytes: f.rd + f.wr, naiveBytes: nv.rd + nv.wr, flashCycles: f.cycles, naiveCycles: nv.cycles });
}
const cmp: any = {};
{ const b = { n: 12, qrows: 4, causal: true, overlap: false, latency: 10 };
  const f = run(toy.q, toy.k, toy.v, { ...b, naive: false }); const nv = run(toy.q, toy.k, toy.v, { ...b, naive: true });
  for (const r of [f, nv]) { (r as any).tileTrace = undefined; (r as any).clocks = undefined; }
  cmp.flash = f; cmp.naive = nv; }
console.log(JSON.stringify(sweepRows));
const out = { sweep: sweepRows, cmp, toy: { labels: toy.labels, cfg: cfgToy, run: toyRun }, n32: { flash, naive, flashOverlap: flashOv } };
// keep it small: drop trace from the n32 runs
for (const r of [flash, naive, flashOv]) { (r as any).tileTrace = undefined; (r as any).clocks = undefined; }
writeFileSync(join(here, "../src/trace.json"), JSON.stringify(out));
console.log("toy", toyRun.cycles, toyRun.tiles, toyRun.rd, toyRun.wr, "tiles traced", toyRun.tileTrace.length);
console.log("n32 flash", flash.rd + flash.wr, flash.cycles, "naive", naive.rd + naive.wr, naive.cycles, "flash+overlap", flashOv.cycles);

// every preset, both schedules, with the app's defaults (qrows 4, latency 10, overlap off, preset's own causal flag)
for (const p of PRESETS) {
  const n = p.q.length;
  const b = { n, qrows: 4, causal: p.causal, overlap: false, latency: 10 };
  const f = run(p.q, p.k, p.v, { ...b, naive: false });
  const nv = run(p.q, p.k, p.v, { ...b, naive: true });
  console.log(p.name, "N", n, "causal", p.causal, "flash B", f.rd + f.wr, "cyc", f.cycles, "tiles", f.tiles, "| naive B", nv.rd + nv.wr, "cyc", nv.cycles, "tiles", nv.tiles);
}
