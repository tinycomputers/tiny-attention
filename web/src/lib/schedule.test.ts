import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { flashTileOrder, naiveTileOrder } from "@/lib/schedule";

const fixture = (name: string) =>
  JSON.parse(readFileSync(join(process.cwd(), "..", "wasm", "fixtures", `${name}.json`), "utf8")) as {
    n: number;
    qrows: number;
    causal: boolean;
    tiles: number;
  };

describe("tile order matches the engine's tile counter", () => {
  it("FLASH sequential", () => {
    const f = fixture("flash_seq");
    expect(flashTileOrder(f.n, f.qrows, f.causal).length).toBe(f.tiles);
  });
  it("FLASH causal with overlap (same order as without)", () => {
    const f = fixture("flash_overlap_causal");
    expect(flashTileOrder(f.n, f.qrows, f.causal).length).toBe(f.tiles);
  });
  it("NAIVE causal", () => {
    const f = fixture("naive_causal");
    expect(naiveTileOrder(f.n, f.causal).length).toBe(f.tiles);
  });
});

describe("tile order invariants", () => {
  it("no tile above the causal diagonal; row groups increase within a K/V tile", () => {
    const order = flashTileOrder(32, 16, true);
    for (const t of order) expect(t.kj).toBeLessThanOrEqual(t.qi);
    for (let i = 1; i < order.length; i++) {
      const a = order[i - 1]!;
      const b = order[i]!;
      if (a.kj === b.kj) {
        expect(b.qi).toBeGreaterThan(a.qi);
      } else if (b.kj > a.kj) {
        // same Q block, next K/V tile
        expect(b.qi).toBeGreaterThanOrEqual(a.qi - 3);
      } else {
        // next Q block starts again at K/V tile 0 with later row groups
        expect(b.kj).toBe(0);
        expect(b.qi).toBeGreaterThan(a.qi);
      }
    }
  });
});
