// Captures the social card and a screenshot of the visualizer from the built site.
//   bash: pnpm build && node scripts/capture-assets.mjs
// Writes public/og.png (1200 x 630, from /og) and docs/media/visualizer.png (from /app, for the README). Rebuild after
// running this so public/og.png is included in dist/.
import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const webRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = join(webRoot, "..");
const port = 4340;

const server = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], {
  cwd: join(webRoot, "dist"),
  stdio: "ignore",
});

async function waitForServer() {
  for (let i = 0; i < 50; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/og/`);
      if (r.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((res) => setTimeout(res, 200));
  }
  throw new Error("static server did not start");
}

try {
  await waitForServer();
  const browser = await chromium.launch();
  try {
    const og = await browser.newPage({ viewport: { width: 1200, height: 630 } });
    await og.goto(`http://127.0.0.1:${port}/og/`);
    await og.waitForLoadState("load");
    await og.screenshot({ path: join(webRoot, "public", "og.png") });

    mkdirSync(join(repoRoot, "docs", "media"), { recursive: true });
    const app = await browser.newPage({ viewport: { width: 1200, height: 900 }, colorScheme: "dark" });
    await app.goto(`http://127.0.0.1:${port}/app/`);
    await app.getByRole("button", { name: "Start" }).click({ timeout: 30_000 });
    await app.getByRole("button", { name: "Run to end" }).click();
    await app.getByRole("heading", { name: "Output" }).waitFor({ timeout: 30_000 });
    await app.screenshot({ path: join(repoRoot, "docs", "media", "visualizer.png"), fullPage: false });
    console.log("wrote web/public/og.png and docs/media/visualizer.png");
  } finally {
    await browser.close();
  }
} finally {
  server.kill();
}
