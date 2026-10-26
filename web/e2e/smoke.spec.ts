import { expect, test } from "@playwright/test";

// The visualizer runs the compiled engine in the page; each test waits for the engine to load first.
async function openApp(page: import("@playwright/test").Page) {
  await page.goto("/app");
  await expect(page.getByRole("button", { name: "Start" })).toBeVisible({ timeout: 30_000 });
}

const clock = (page: import("@playwright/test").Page) =>
  page.getByTestId("clock").textContent().then((t) => Number(t));

test("FLASH run to the end: output matches the float reference", async ({ page }) => {
  await openApp(page);
  await page.getByRole("button", { name: "Start" }).click();
  await page.getByRole("button", { name: "Run to end" }).click();

  await expect(page.getByRole("heading", { name: "Output" })).toBeVisible();
  const text = await page.locator("main").innerText();
  const match = text.match(/Max \|engine − float attention\| = ([0-9.]+)/);
  expect(match, "max deviation line").not.toBeNull();
  expect(Number(match?.[1])).toBeLessThan(0.02);
  await expect(page.getByText("Float reference attention")).toBeVisible();
  await expect(page.getByText(/· in use/)).toBeVisible();
  await expect(page.getByText(/ B$/).first()).toBeVisible();
});

test("step forward and back moves the clock", async ({ page }) => {
  await openApp(page);
  await page.getByRole("button", { name: "Start" }).click();
  expect(await clock(page)).toBe(0);
  for (let i = 0; i < 3; i++) await page.getByRole("button", { name: "Step ▶" }).click();
  expect(await clock(page)).toBe(3);
  await page.getByRole("button", { name: "◀ Back" }).click();
  expect(await clock(page)).toBe(2);
});

test("NAIVE run shows the engine's probability matrix from HBM", async ({ page }) => {
  await openApp(page);
  await page.getByLabel("Schedule").selectOption("1");
  await page.getByRole("button", { name: "Start" }).click();
  await page.getByRole("button", { name: "Run to end" }).click();

  await expect(page.getByText("read back from HBM (NAIVE)")).toBeVisible();
  await expect(page.getByText("Read", { exact: true })).toBeVisible();
});

test("editing a Q value resets the run and the result follows the new input", async ({ page }) => {
  await openApp(page);
  await page.getByRole("button", { name: "Start" }).click();
  await page.getByRole("button", { name: "Run to end" }).click();
  await expect(page.getByRole("heading", { name: "Output" })).toBeVisible();

  await page.getByText(/Edit Q, K and V/).click();
  const cell = page.getByLabel("Q token 0 dimension 0");
  await cell.fill("2.5");
  expect(await clock(page)).toBe(0);
  await expect(page.getByRole("heading", { name: "Output" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Reset to preset" })).toBeEnabled();

  // the engine still matches the float reference for the edited inputs
  await page.getByRole("button", { name: "Start" }).click();
  await page.getByRole("button", { name: "Run to end" }).click();
  const text = await page.locator("main").innerText();
  const match = text.match(/Max \|engine − float attention\| = ([0-9.]+)/);
  expect(Number(match?.[1])).toBeLessThan(0.02);
});

test("no horizontal scroll at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await openApp(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
