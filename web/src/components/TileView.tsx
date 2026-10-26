// L2: the datapath of one (qi, kj) step, with the live wire values at the current clock. The two arrays are PE grids:
// PE (r, c) holds the value index r*4 + c. Values are the engine's; the grids show what the wires carry now.
import type { ReactNode } from "react";
import type { Datapath, TileState } from "@/lib/state-schema";

interface Props {
  datapath: Datapath | null;
  tile: TileState | null;
  naive: boolean;
}

const muted = { color: "var(--muted)" } as const;

function Grid({ title, values, scale = 1, digits = 0 }: { title: string; values: number[]; scale?: number; digits?: number }) {
  const cells = values.slice(0, 16);
  return (
    <div>
      <div className="text-[10px]" style={muted}>{title}</div>
      <div className="overflow-x-auto">
        <div className="inline-grid gap-x-2 font-mono text-[11px] tabular-nums" style={{ gridTemplateColumns: "repeat(4, auto)" }}>
          {cells.map((v, i) => <span key={i}>{(v * scale).toFixed(digits)}</span>)}
        </div>
      </div>
    </div>
  );
}

function Stage({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div className="rounded p-3" style={{ background: "var(--panel)", border: "1px solid var(--line)" }}>
      <div className="mb-2 text-xs font-semibold">{name}</div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

export default function TileView({ datapath, tile, naive }: Props) {
  if (naive) {
    return (
      <p className="text-sm" style={muted}>
        NAIVE runs its passes through the same arrays, but its Z and P tiles go out to HBM between passes. Switch to FLASH
        to see the single-tile datapath.
      </p>
    );
  }
  if (!datapath || !tile) {
    return <p className="text-sm" style={muted}>Run the engine to fill this view.</p>;
  }
  return (
    <div className="space-y-4">
      <p className="text-xs" style={muted}>
        Wires at the last busy clock of the run so far (between tiles they are idle). The S, Z, P and O grids are for the row group of the most recent tile.
      </p>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        <Stage name="MM1: S = Q · Kᵀ (array)">
          <Grid title="Kᵀ weights (int8)" values={datapath.mm1W} />
          <Grid title="Q activations (int8)" values={datapath.mm1A} />
          <Grid title="S partial sums (int17)" values={datapath.mm1Psum} />
        </Stage>
        <Stage name="Scale and mask">
          <Grid title="Z = S · scale (Q8.8, log2)" values={tile.z} scale={1 / 256} digits={2} />
        </Stage>
        <Stage name="Online softmax (row max, exp2, sum)">
          <div className="grid grid-cols-3 gap-2">
            <Grid title="m" values={tile.m} scale={1 / 256} digits={2} />
            <Grid title="α" values={tile.alpha} scale={1 / 32768} digits={3} />
            <Grid title="l" values={tile.l} scale={1 / 4096} digits={2} />
          </div>
          <Grid title="P = 2^(Z − m) (UQ1.8)" values={tile.p} scale={1 / 256} digits={3} />
        </Stage>
        <Stage name="MM2: P · V (array)">
          <Grid title="V weights (int8)" values={datapath.mm2W} />
          <Grid title="P activations (uint9)" values={datapath.mm2A} scale={1 / 256} digits={3} />
          <Grid title="P·V partial sums (int18)" values={datapath.mm2Psum} />
        </Stage>
        <Stage name="O accumulator (rescaled by α)">
          <Grid title="O (Q8.8 accumulator, V-LSB)" values={tile.o} scale={1 / 256} digits={1} />
        </Stage>
      </div>
    </div>
  );
}
