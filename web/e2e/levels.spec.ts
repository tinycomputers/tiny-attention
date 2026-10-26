import { expect, test, type Page } from "@playwright/test";

async function openApp(page: Page) {
  await page.goto("/app");
  await expect(page.getByRole("button", { name: "Start" })).toBeVisible({ timeout: 30_000 });
}

test("L1 softmax cell: running max and sum for the chosen row, against the float reference", async ({ page }) => {
  await openApp(page);
  await page.getByRole("button", { name: "Start" }).click();
  await page.getByRole("button", { name: "Run to end" }).click();
  await page.getByRole("tab", { name: "L1 Softmax cell" }).click();

  await expect(page.getByText("m engine / float")).toBeVisible();
  await expect(page.getByRole("cell", { name: "kv0" })).toBeVisible();
  const text = await page.locator("main").innerText();
  const m = text.match(/m engine \/ float\n([0-9.-]+) \/ ([0-9.-]+)/);
  expect(m, "m engine / float line").not.toBeNull();
  expect(Math.abs(Number(m?.[1]) - Number(m?.[2]))).toBeLessThan(0.05);
});

test("L1 says so for NAIVE", async ({ page }) => {
  await openApp(page);
  await page.getByRole("combobox", { name: /^Schedule/ }).selectOption("1");
  await page.getByRole("tab", { name: "L1 Softmax cell" }).click();
  await expect(page.getByText("NAIVE has no running max")).toBeVisible();
});

test("L2 tile dataflow shows the MM1 and MM2 arrays with live values", async ({ page }) => {
  await openApp(page);
  await page.getByRole("button", { name: "Start" }).click();
  await page.getByRole("button", { name: "Run to end" }).click();
  await page.getByRole("tab", { name: "L2 Tile dataflow" }).click();

  await expect(page.getByText("MM1: S = Q · Kᵀ (array)")).toBeVisible();
  await expect(page.getByText("MM2: P · V (array)")).toBeVisible();
  await expect(page.getByText("O accumulator (rescaled by α)")).toBeVisible();
});

test("L4 runs both schedules on the same inputs and sweeps N", async ({ page }) => {
  await openApp(page);
  await page.getByRole("tab", { name: "L4 Naive vs Flash" }).click();
  await page.getByRole("button", { name: "Run both on these inputs" }).click();
  await expect(page.getByText("Same inputs, both schedules")).toBeVisible();
  await expect(page.getByText(/FLASH, max [0-9.]+, mean [0-9.]+/)).toBeVisible();

  await page.getByRole("button", { name: "Sweep N = 4 to 32" }).click();
  await expect(page.getByText("HBM bytes against N (seeded inputs, same settings)")).toBeVisible();
  // header row plus one row per N: 4, 8, ..., 32
  const sweep = page.locator("table").last();
  await expect(sweep.locator("tbody tr")).toHaveCount(8);
});

test("light and dark colour schemes both render the visualizer", async ({ browser }) => {
  for (const scheme of ["light", "dark"] as const) {
    const context = await browser.newContext({ colorScheme: scheme });
    const page = await context.newPage();
    await openApp(page);
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    expect(bg).not.toBe("rgba(0, 0, 0, 0)");
    await context.close();
  }
});

test("L4 at phone width: no horizontal scroll", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await openApp(page);
  await page.getByRole("tab", { name: "L4 Naive vs Flash" }).click();
  await page.getByRole("button", { name: "Run both on these inputs" }).click();
  await expect(page.getByText("Same inputs, both schedules")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test("every view has no horizontal scroll at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await openApp(page);
  await page.getByRole("button", { name: "Start" }).click();
  await page.getByRole("button", { name: "Run to end" }).click();
  for (const name of ["L1 Softmax cell", "L2 Tile dataflow", "L3 Schedule", "L4 Naive vs Flash"]) {
    await page.getByRole("tab", { name }).click();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, name).toBeLessThanOrEqual(0);
  }
});
