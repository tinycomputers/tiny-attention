import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { quantize, roundHalfEven } from "@/lib/quantize";

const cases = JSON.parse(
  readFileSync(join(process.cwd(), "..", "wasm", "fixtures", "quantize_cases.json"), "utf8"),
) as Array<{
  q: number[][];
  k: number[][];
  v: number[][];
  expQ: number[];
  expK: number[];
  expV: number[];
  expCfgScale: number;
}>;

describe("roundHalfEven", () => {
  it("matches numpy rint / Python round on halves", () => {
    expect([0.5, 1.5, 2.5, -0.5, -1.5, 2.4999, 2.5001].map(roundHalfEven)).toEqual([0, 2, 2, 0, -2, 2, 3]);
  });
});

describe("quantize (mirror of sim/golden/fixed.py)", () => {
  cases.forEach((c, i) => {
    it(`case ${i} matches the golden values exactly`, () => {
      const z = quantize(c.q, c.k, c.v);
      expect(z.q).toEqual(c.expQ);
      expect(z.k).toEqual(c.expK);
      expect(z.v).toEqual(c.expV);
      expect(z.cfgScale).toBe(c.expCfgScale);
    });
  });

  it("keeps values inside int8 and cfg_scale inside 24 bits", () => {
    for (const c of cases) {
      const z = quantize(c.q, c.k, c.v);
      for (const x of [...z.q, ...z.k, ...z.v]) expect(Math.abs(x)).toBeLessThanOrEqual(127);
      expect(z.cfgScale).toBeLessThan(2 ** 24);
    }
  });
});
