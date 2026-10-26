import { defineConfig, devices } from "@playwright/test";

// Runs against the production build (`pnpm test:e2e` builds first).
export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:4325",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "pnpm exec astro preview --port 4325",
    url: "http://localhost:4325/app",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
