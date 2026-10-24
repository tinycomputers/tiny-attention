import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwindcss from "@tailwindcss/vite";
import { createReadStream, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Dev-server middleware that serves the generated Emscripten .mjs files from public/
 * as raw static JS. Vite 6 rejects dynamic imports of public/ files (it appends ?import and
 * 500s), so we short-circuit module resolution for these paths. No effect in production.
 */
const wasmJsPassthrough = {
  name: "wasm-js-passthrough",
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const url = (req.url ?? "").split("?")[0];
      if (url === "/tiny_attn.mjs" || url === "/spike/counter.mjs") {
        const filePath = join(__dirname, "public", url);
        if (existsSync(filePath)) {
          res.setHeader("Content-Type", "application/javascript; charset=utf-8");
          res.setHeader("Cache-Control", "no-cache");
          createReadStream(filePath).pipe(res);
          return;
        }
      }
      next();
    });
  },
};

export default defineConfig({
  site: "https://tiny-attention.vercel.app",
  integrations: [react()],
  vite: {
    plugins: [tailwindcss(), wasmJsPassthrough],
    build: {
      rollupOptions: {
        // Served from public/ at runtime; Rollup must not try to bundle them.
        external: ["/tiny_attn.mjs", "/spike/counter.mjs"],
      },
    },
  },
});
