import { bundle } from "@remotion/bundler";
import { renderStill, selectComposition } from "@remotion/renderer";
import path from "node:path";
const frames = process.argv.slice(2).map(Number);
const browserExecutable = "/home/deaneeth/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell";
const serveUrl = await bundle({ entryPoint: path.resolve("src/index.ts"), onProgress: () => {} });
const comp = await selectComposition({ serveUrl, id: "Demo", browserExecutable, chromiumOptions: { gl: "angle" }, timeoutInMilliseconds: 180000 });
for (const f of frames) {
  await renderStill({ composition: comp, serveUrl, frame: f, output: `out/check/f${f}.png`, browserExecutable, chromiumOptions: { gl: "angle" }, timeoutInMilliseconds: 180000, imageFormat: "png", overwrite: true });
  console.log("done", f);
}
