import { continueRender, delayRender, staticFile } from "remotion";

const load = async () => {
  const faces = [
    new FontFace("Inter Variable", `url(${staticFile("fonts/inter-latin-wght-normal.woff2")})`, { weight: "100 900" }),
    new FontFace("JetBrains Mono Variable", `url(${staticFile("fonts/jetbrains-mono-latin-wght-normal.woff2")})`, { weight: "100 800" }),
  ];
  await Promise.all(faces.map((f) => f.load()));
  faces.forEach((f) => document.fonts.add(f));
};
const h = delayRender("fonts");
load().then(() => continueRender(h)).catch((e) => { console.error(e); continueRender(h); });
