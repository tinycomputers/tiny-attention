import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { C, EASE, FONT, clamp } from "../theme";
import { Enter, Mark } from "../components/bits";

export const CTA_CLICK = 206; // local frame the button is pressed
const STEPS = [
  { t: "Pick a preset", at: 52 },
  { t: "Press Run to end", at: 74 },
  { t: "Watch every tile resolve", at: 94 },
];

export const Cta: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const fadeOut = interpolate(f, [dur - 14, dur], [0, 1], { ...clamp, easing: EASE.in });
  const press = interpolate(f, [CTA_CLICK, CTA_CLICK + 3, CTA_CLICK + 12], [1, 0.97, 1], { ...clamp, easing: EASE.out });
  const rip = interpolate(f, [CTA_CLICK, CTA_CLICK + 24], [0, 1], { ...clamp, easing: EASE.out });
  const cx = interpolate(f, [170, CTA_CLICK - 4], [1300, 1000], { ...clamp, easing: EASE.inOut });
  const cy = interpolate(f, [170, CTA_CLICK - 4], [860, 722], { ...clamp, easing: EASE.inOut });
  const cv = interpolate(f, [164, 176, CTA_CLICK + 40, CTA_CLICK + 52], [0, 1, 1, 0], clamp);
  return (
    <AbsoluteFill style={{ opacity: 1 - fadeOut }}>
      <div style={{ position: "absolute", left: 0, right: 0, top: 110, display: "flex", justifyContent: "center" }}>
        <Enter y={12}>
          <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
            <Mark size={60} />
            <span style={{ fontFamily: FONT.sans, fontWeight: 700, fontSize: 42, letterSpacing: "-0.03em", color: C.white }}>Tiny<span style={{ color: C.accent }}>Attention</span></span>
          </div>
        </Enter>
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: 262, textAlign: "center", fontFamily: FONT.sans, fontWeight: 720, fontSize: 140, letterSpacing: "-0.045em", lineHeight: 1, color: C.white }}>
        <Enter delay={6} y={40} style={{ display: "inline-block", marginRight: 30 }}>Try it</Enter>
        <Enter delay={14} y={44} cfg="snappy" style={{ display: "inline-block", color: C.accent }}>yourself.</Enter>
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: 520, display: "flex", justifyContent: "center", gap: 28 }}>
        {STEPS.map((s, i) => (
          <Enter key={i} delay={s.at} y={20} cfg="snappy">
            <div style={{ width: 460, whiteSpace: "nowrap", borderTop: `1px solid ${C.line}`, padding: "20px 2px", display: "flex", gap: 18, alignItems: "baseline" }}>
              <span style={{ fontFamily: FONT.mono, fontSize: 20, color: C.accent }}>{`0${i + 1}`}</span>
              <span style={{ fontFamily: FONT.sans, fontSize: 30, fontWeight: 560, color: C.white }}>{s.t}</span>
            </div>
          </Enter>
        ))}
      </div>
      <Enter delay={22} y={30} cfg="pop" style={{ position: "absolute", left: 0, right: 0, top: 690, display: "flex", justifyContent: "center" }}>
        <div style={{ position: "relative", transform: `scale(${press})` }}>
          <div style={{ position: "absolute", inset: -6, borderRadius: 16, border: `2px solid ${C.accent}`, opacity: 1 - rip, transform: `scale(${1 + rip * 0.12})` }} />
          <div style={{ background: C.accent, color: C.bg, fontFamily: FONT.sans, fontWeight: 700, fontSize: 40, letterSpacing: "-0.02em", padding: "22px 56px", borderRadius: 10, display: "flex", alignItems: "center", gap: 18 }}>
            Open the engine
            <svg width="34" height="34" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6" stroke={C.bg} strokeWidth="2.6" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </div>
        </div>
      </Enter>
      <div style={{ position: "absolute", left: cx, top: cy, opacity: cv, zIndex: 9 }}>
        <svg width="46" height="54" viewBox="0 0 24 28" style={{ filter: "drop-shadow(0 4px 8px #000a)" }}>
          <path d="M2 2 L2 21 L7 16.5 L10.5 25 L13.5 23.7 L10 15.5 L17 15.5 Z" fill="#fff" stroke="#0d1117" strokeWidth="1.6" strokeLinejoin="round" />
        </svg>
      </div>
      <Enter delay={190} y={12} style={{ position: "absolute", left: 0, right: 0, top: 880, textAlign: "center" }}>
        <div style={{ fontFamily: FONT.sans, fontSize: 28, fontWeight: 430, color: C.muted }}>Runs in your browser. Nothing to install.</div>
      </Enter>
    </AbsoluteFill>
  );
};
