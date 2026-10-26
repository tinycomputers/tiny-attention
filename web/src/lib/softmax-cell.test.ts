import { describe, expect, it } from "vitest";
import { rowReference, rowSteps, type TileSample } from "@/lib/softmax-cell";
import { seededInputs } from "@/lib/engine-run";

const q = seededInputs(8, 7).q;
const k = seededInputs(8, 7).k;

describe("rowReference (float, same INT8 inputs)", () => {
  it("causal row 0 sees only key 0: m is its logit and l is 1", () => {
    const r = rowReference(q, k, 0, true);
    expect(r.z.slice(1).every((x) => x === -Infinity)).toBe(true);
    expect(r.m).toBe(r.z[0]);
    expect(r.l).toBeCloseTo(1, 12);
  });

  it("non-causal: l is the sum of 2^(z - m) and is at least 1", () => {
    const r = rowReference(q, k, 3, false);
    expect(r.z.every((x) => Number.isFinite(x))).toBe(true);
    expect(r.l).toBeGreaterThanOrEqual(1);
    const direct = r.z.reduce((acc, x) => acc + 2 ** (x - r.m), 0);
    expect(r.l).toBeCloseTo(direct, 12);
  });
});

describe("rowSteps", () => {
  const sample = (qi: number, kj: number, base: number): TileSample => ({
    qi,
    kj,
    z: Array.from({ length: 16 }, (_, i) => (base + i) * 256),
    m: [base * 256, 0, 0, 0],
    l: [base * 4096, 0, 0, 0],
    alpha: [32768, 0, 0, 0],
  });

  it("keeps only the tiles of the row's group and reads its lane", () => {
    const samples = [sample(0, 0, 1), sample(1, 0, 5), sample(0, 1, 2)];
    const steps = rowSteps(samples, 0);
    expect(steps.map((s) => s.kj)).toEqual([0, 1]);
    expect(steps[0]?.z).toEqual([1, 2, 3, 4]);
    expect(steps[1]?.m).toBe(2);
    expect(steps[0]?.alpha).toBe(1);
  });

  it("row 5 is lane 1 of group 1", () => {
    const steps = rowSteps([sample(1, 0, 5)], 5);
    expect(steps[0]?.z).toEqual([5 + 4, 5 + 5, 5 + 6, 5 + 7]);
  });
});

describe("seededInputs", () => {
  it("is deterministic and has the requested shape and bounds", () => {
    const a = seededInputs(12, 99);
    const b = seededInputs(12, 99);
    expect(a).toEqual(b);
    expect(a.q.length).toBe(12);
    expect(a.q[0]?.length).toBe(4);
    expect(a.q.flat().every((x) => Math.abs(x) <= 2)).toBe(true);
    expect(a.v.flat().every((x) => Math.abs(x) <= 1)).toBe(true);
  });
});
