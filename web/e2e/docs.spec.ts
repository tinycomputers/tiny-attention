import { expect, test, type Page } from "@playwright/test";

const pages = [
  { path: "/", h1: "FlashAttention, built in hardware" },
  { path: "/docs", h1: "Docs" },
  { path: "/docs/what-is-attention", h1: "What is attention" },
  { path: "/docs/online-softmax", h1: "Online softmax" },
  { path: "/docs/flashattention", h1: "FlashAttention" },
  { path: "/docs/fixed-point", h1: "Fixed point" },
  { path: "/docs/architecture", h1: "Architecture" },
  { path: "/docs/verification", h1: "Verification" },
];

// Collects console errors and page errors so a broken island fails the test.
function watchErrors(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  return errors;
}

for (const p of pages) {
  test(`${p.path} renders with no errors`, async ({ page }) => {
    const errors = watchErrors(page);
    const response = await page.goto(p.path);
    expect(response?.status()).toBe(200);
    await expect(page.getByRole("heading", { level: 1 })).toContainText(p.h1);
    await page.waitForTimeout(500);
    expect(errors, errors.join("\n")).toEqual([]);
  });
}

test("landing page links to the engine and the docs", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("link", { name: "Open the engine" })).toHaveAttribute("href", "/app");
  await expect(page.getByRole("link", { name: "Read the docs" })).toHaveAttribute("href", "/docs");
});

test("landing hero animates the tile schedule", async ({ page }) => {
  await page.goto("/");
  const grid = page.getByRole("img", { name: /FLASH tile schedule/ });
  await expect(grid).toBeVisible();
  const before = await grid.evaluate((g) => g.innerHTML);
  await page.waitForTimeout(700);
  const after = await grid.evaluate((g) => g.innerHTML);
  expect(after).not.toEqual(before);
});

test("what is attention: the demo's probabilities sum to 100%", async ({ page }) => {
  await page.goto("/docs/what-is-attention");
  await expect(page.getByText("Token that asks (query)")).toBeVisible();
  const text = await page.locator("main").innerText();
  const shares = [...text.matchAll(/(\d+\.\d)%/g)].map((m) => Number(m[1]));
  expect(shares.length).toBeGreaterThan(0);
  const sum = shares.reduce((a, b) => a + b, 0);
  expect(Math.abs(sum - 100)).toBeLessThan(1);
});

test("online softmax: the engine records a run and the row table fills", async ({ page }) => {
  await page.goto("/docs/online-softmax");
  await expect(page.getByRole("cell", { name: "kv0" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("m engine / float")).toBeVisible();
});

test("flashattention: traffic demo shows FLASH and NAIVE byte counts", async ({ page }) => {
  await page.goto("/docs/flashattention");
  await expect(page.getByText(/FLASH \(Q block 4 rows\)/)).toBeVisible();
  await expect(page.getByText(/NAIVE \(N×N scores in HBM\)/)).toBeVisible();
  // at N = 32 with causal off, FLASH with 4-row blocks moves a different total than NAIVE
  await page.getByLabel("N").selectOption("32");
  await expect(page.getByText(/FLASH moves [0-9.]+× fewer bytes than NAIVE here/)).toBeVisible();
});

test("fixed point: the quantizer shows the INT8 values and cfg_scale in hex", async ({ page }) => {
  await page.goto("/docs/fixed-point");
  await expect(page.getByText(/= \d+ = 0x[0-9a-f]{6}/)).toBeVisible();
  await page.getByLabel("q dimension 0").fill("3.5");
  await expect(page.getByText(/int8 127/).first()).toBeVisible();
});

for (const p of pages) {
  test(`${p.path} has no horizontal scroll at phone width`, async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto(p.path);
    await page.waitForTimeout(300);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });
}

test("social metadata is present on the landing page", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator('meta[property="og:image"]')).toHaveCount(1);
  await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute("content", "summary_large_image");
});
