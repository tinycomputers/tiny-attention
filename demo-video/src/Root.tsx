import React from "react";
import { AbsoluteFill, Composition, Sequence, interpolate, staticFile } from "remotion";
import { Audio } from "@remotion/media";
import "./fonts";
import { FPS, H, W, C, clamp } from "./theme";
import { SCENES, TOTAL, SceneId } from "./timeline";
import { Backdrop } from "./components/Backdrop";
import { Hook } from "./scenes/Hook";
import { Intro } from "./scenes/Intro";
import { AppStage, CLICK_FRAMES, END } from "./scenes/AppStage";
import { Cta, CTA_CLICK } from "./scenes/Cta";

const IDS: SceneId[] = ["hook", "intro", "tiles", "engine", "memory", "trust", "cta"];

// Music sits low and steps down further while the narrator speaks.
const duck = (g: number) => {
  let d = 0;
  for (const id of IDS) {
    const s = SCENES[id];
    d = Math.max(d, interpolate(g, [s.voFrom - 10, s.voFrom, s.voFrom + s.voDur, s.voFrom + s.voDur + 14], [0, 1, 1, 0], clamp));
  }
  return 0.3 - 0.14 * d;
};

const sfx = (name: string, at: number, vol: number, key: string) => (
  <Sequence key={key} from={Math.max(0, at)} durationInFrames={90} layout="none">
    <Audio src={staticFile(`sfx/${name}.wav`)} volume={vol} />
  </Sequence>
);

const Demo: React.FC = () => {
  const tilesFrom = SCENES.tiles.from;
  const sound: React.ReactNode[] = [];
  (["intro", "tiles", "engine", "memory", "trust"] as SceneId[]).forEach((id) => sound.push(sfx("whoosh", SCENES[id].from - 4, 0.35, `w-${id}`)));
  sound.push(sfx("hit", 24, 0.55, "hit-problem"));
  CLICK_FRAMES.forEach((c, i) => sound.push(sfx("pop", tilesFrom + c - 1, 0.35, `pop-${i}`)));
  for (let i = 0; i < 24; i++) sound.push(sfx("tick", tilesFrom + 52 + i * 8, 0.1, `tick-${i}`));
  sound.push(sfx("riser", SCENES.cta.from - 46, 0.4, "riser"));
  sound.push(sfx("hit", SCENES.cta.from, 0.6, "hit-cta"));
  sound.push(sfx("pop", SCENES.cta.from + CTA_CLICK - 1, 0.5, "pop-cta"));
  sound.push(sfx("chime", SCENES.cta.from + CTA_CLICK, 0.3, "chime"));

  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <Backdrop />
      <Sequence from={SCENES.hook.from} durationInFrames={SCENES.hook.dur}><Hook dur={SCENES.hook.dur} /></Sequence>
      <Sequence from={SCENES.intro.from} durationInFrames={SCENES.intro.dur}><Intro dur={SCENES.intro.dur} /></Sequence>
      <Sequence from={tilesFrom} durationInFrames={END}><AppStage /></Sequence>
      <Sequence from={SCENES.cta.from} durationInFrames={SCENES.cta.dur}><Cta dur={SCENES.cta.dur} /></Sequence>

      <Audio src={staticFile("audio/music.wav")} volume={(g) => duck(g) * interpolate(g, [TOTAL - 90, TOTAL], [1, 0], clamp)} />
      {IDS.map((id) => (
        <Sequence key={`vo-${id}`} from={SCENES[id].voFrom} layout="none"><Audio src={staticFile(`audio/vo_${id}.wav`)} volume={1} /></Sequence>
      ))}
      {sound}
    </AbsoluteFill>
  );
};

export const Root: React.FC = () => (
  <Composition id="Demo" component={Demo} durationInFrames={TOTAL} fps={FPS} width={W} height={H} />
);
