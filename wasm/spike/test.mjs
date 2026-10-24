// Headless check of the spike build: two instances of one model count independently, at
// different speeds, with their steps interleaved.
// Usage: node wasm/spike/test.mjs   (after bash wasm/spike/build.sh)
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const dir = join(dirname(fileURLToPath(import.meta.url)), "../../web/public/spike");
const { default: create } = await import(join(dir, "counter.mjs"));
const mod = await create();

const a = new mod.CounterSim();
const b = new mod.CounterSim();
a.reset();
b.reset();
const ra = [];
const rb = [];
// A runs 3x as fast as B; stepping one must never move the other.
for (let i = 1; i <= 6; i++) {
  a.step();
  ra.push(a.getCount());
  if (i % 3 === 0) {
    b.step();
    rb.push(b.getCount());
  }
}
// Resetting B must not disturb A.
b.reset();
const afterReset = { a: a.getCount(), b: b.getCount() };
a.delete();
b.delete();

const ok =
  ra.join() === "1,2,3,4,5,6" && rb.join() === "1,2" && afterReset.a === 6 && afterReset.b === 0;
console.log(`A: ${ra}  B: ${rb}  after B.reset: ${JSON.stringify(afterReset)}  ->  ${ok ? "PASS" : "FAIL"}`);
process.exit(ok ? 0 : 1);
