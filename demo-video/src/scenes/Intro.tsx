import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { C, EASE, FONT, clamp, sp } from "../theme";
import { Enter, Mark } from "../components/bits";

const NODES = [
  { k: "rtl/*.sv", t: "SystemVerilog", d: "Two INT8 systolic arrays and an online softmax unit" },
  { k: "Verilator", t: "C++ model", d: "Cycle-accurate simulation of the real RTL" },
  { k: "Emscripten", t: "tiny_attn.wasm", d: "Under 100 KB gzipped" },
  { k: "React + SVG", t: "Your browser", d: "Step it clock by clock" },
];

export const Intro: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const ex = interpolate(f, [dur - 10, dur], [0, 1], { ...clamp, easing: EASE.in });
  const move = interpolate(f, [88, 128], [0, 1], { ...clamp, easing: EASE.inOut });
  const markP = interpolate(f, [4, 34], [0, 1], { ...clamp, easing: EASE.out });
  const breathe = 1 + Math.sin(f / 14) * 0.012;
  const word = "TinyAttention".split("");
  return (
    <AbsoluteFill style={{ opacity: 1 - ex, transform: `scale(${1 - ex * 0.03})` }}>
      <div style={{ position: "absolute", left: 0, right: 0, top: interpolate(move, [0, 1], [372, 150]), display: "flex", flexDirection: "column", alignItems: "center", transform: `scale(${interpolate(move, [0, 1], [1, 0.72])})`, transformOrigin: "50% 0" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 40, transform: `scale(${breathe})` }}>
          <Mark size={170} progress={markP} glow={markP} />
          <div style={{ fontFamily: FONT.sans, fontWeight: 800, fontSize: 168, letterSpacing: "-0.05em", color: C.white, display: "flex", lineHeight: 1 }}>
            {word.map((ch, i) => {
              const a = sp(f, fps, 18 + i * 1.6, "snappy");
              return <span key={i} style={{ display: "inline-block", opacity: a, transform: `translateY(${(1 - a) * 44}px)`, color: i >= 4 ? C.accent : C.white, textShadow: "none" }}>{ch}</span>;
            })}
          </div>
        </div>
        <Enter delay={46} y={22} style={{ marginTop: 26 }}>
          <div style={{ fontFamily: FONT.sans, fontSize: 40, fontWeight: 450, color: C.muted, letterSpacing: "-0.01em" }}>
            FlashAttention, built in hardware and run in your browser.
          </div>
        </Enter>
      </div>
      {/* pipeline */}
      <div style={{ position: "absolute", left: 110, right: 110, top: 530, display: "flex", alignItems: "stretch", justifyContent: "space-between" }}>
        {NODES.map((n, i) => {
          const d = 104 + i * 9;
          const lit = interpolate((f - 130) % 120, [i * 22, i * 22 + 10, i * 22 + 26], [0, 1, 0], clamp);
          return (
            <React.Fragment key={i}>
              <Enter delay={d} y={50} cfg="smooth" style={{ width: 360 }}>
                <div style={{ background: C.panel, border: `1px solid ${lit > 0.2 && f > 130 ? C.accent : C.line}`, borderRadius: 16, padding: "26px 28px", height: 214, boxSizing: "border-box", boxShadow: "none" }}>
                  <div style={{ fontFamily: FONT.mono, fontSize: 22, color: C.accent, fontWeight: 600 }}>{n.k}</div>
                  <div style={{ fontFamily: FONT.sans, fontSize: 38, fontWeight: 720, color: C.white, marginTop: 10, letterSpacing: "-0.02em" }}>{n.t}</div>
                  <div style={{ fontFamily: FONT.sans, fontSize: 24, color: C.muted, marginTop: 10, lineHeight: 1.35 }}>{n.d}</div>
                </div>
              </Enter>
              {i < NODES.length - 1 && (
                <div style={{ flex: 1, position: "relative", alignSelf: "center", height: 4, margin: "0 6px", opacity: interpolate(f, [d + 8, d + 20], [0, 1], clamp) }}>
                  <div style={{ position: "absolute", inset: 0, background: C.line, borderRadius: 2 }} />
                  <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${interpolate(f, [d + 8, d + 24], [0, 100], { ...clamp, easing: EASE.out })}%`, background: C.accent, opacity: 0.55, borderRadius: 2 }} />
                  <div style={{ position: "absolute", top: -6, width: 16, height: 16, borderRadius: 8, background: C.accent, boxShadow: "none", left: `${((f * 1.6 + i * 30) % 100)}%`, opacity: f > 130 ? 1 : 0 }} />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
