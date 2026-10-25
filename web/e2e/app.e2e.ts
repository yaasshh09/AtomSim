import { expect, test, type Page } from "@playwright/test";

/** Console errors collected per test: a clean page is part of passing. */
function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  return errors;
}

test("the 2p cross-section is drawn from the physics: a dark nodal plane, bright lobes", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/?n=2&l=1&m=0&view=plane");
  await expect(page.locator(".view-header .badge", { hasText: "EXACT" }).first()).toBeVisible({
    timeout: 30_000,
  });
  // Visual regression on the raster itself, not a screenshot: fonts differ by
  // OS, the physics must not. The canvas is the engine's grid, row 0 at +z.
  const px = await page.locator("canvas.plane-raster").evaluate((c: HTMLCanvasElement) => {
    const ctx = c.getContext("2d")!;
    const at = (x: number, y: number) => Array.from(ctx.getImageData(x, y, 1, 1).data.slice(0, 3));
    const mid = c.width / 2;
    // 2p0 peaks at |z| = 2 bohr on the axis; the frame's half-extent is 10 bohr.
    const lobe = Math.round(mid - (2 / 10) * mid);
    return { node: at(mid, mid), lobe: at(mid, lobe), mirror: at(mid, c.height - 1 - lobe) };
  });
  expect(Math.max(...px.node)).toBeLessThan(40); // inferno near zero: almost black
  expect(px.lobe[0]).toBeGreaterThan(230); // inferno near its top: [252, 255, 164]
  expect(px.lobe[1]).toBeGreaterThan(230);
  expect(px.mirror).toEqual(px.lobe); // |psi|^2 is even under z -> -z
  expect(errors).toEqual([]);
});

test("a deep link keeps the state it names", async ({ page }) => {
  await page.goto("/?n=3&l=2&m=-1&system=mu-h&view=levels");
  await expect(page.locator(".state-card-title")).toContainText("3d");
  await expect(page).toHaveURL(/n=3/);
  await expect(page).toHaveURL(/system=mu-h/);
});

test("the hydrogen ladder starts at the ground state", async ({ page }) => {
  await page.goto("/?view=levels");
  await expect(page.getByText(/−13\.6\d* eV/).first()).toBeVisible({ timeout: 30_000 });
});

test("the 3-D cloud samples and says how many points it drew", async ({ page }) => {
  await page.goto("/?n=2&l=1&m=0&view=cloud");
  await expect(page.getByText(/points drawn/)).toBeVisible({ timeout: 60_000 });
});

test("the guide, the changelog and the generated reference all load", async ({ page }) => {
  await page.goto("/api.html");
  await expect(page.locator("h1")).toHaveText("The AtomSim API");
  await page.goto("/changelog.html");
  await expect(page.locator("h1")).toHaveText("What's new");
  const docs = await page.request.get("/openapi.json");
  expect(docs.ok()).toBe(true);
});

test("the nuclear-mass slider walks hydrogen to positronium and flags the made-up nuclei", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/?view=levels");
  const slider = page.locator(".mass-slider input[type=range]");
  await expect(slider).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".mass-slider")).toContainText("hydrogen, a real atom");
  // A mass no particle has: its own key, and the badge on mu says counterfactual.
  await slider.fill("2.699"); // log10(500) on the 0.001 step grid
  await expect(page).toHaveURL(/system=m500(&|$)/, { timeout: 10_000 });
  await expect(page.locator(".mass-slider .badge", { hasText: /counterfactual/i })).toBeVisible();
  await expect(page.locator(".mass-slider")).toContainText("no such nucleus exists");
  await expect(page.getByText(/−13\.5\d* eV/).first()).toBeVisible({ timeout: 30_000 });
  // The bottom of the dial snaps onto the real positronium preset: half of hydrogen.
  await slider.fill("0");
  await expect(page).toHaveURL(/system=ps(&|$)/, { timeout: 10_000 });
  await expect(page.getByText(/−6\.80\d* eV/).first()).toBeVisible({ timeout: 30_000 });
  // And a deep link to a mass key opens on it.
  await page.goto("/?system=m206.768&view=levels");
  await expect(page.locator(".mass-slider")).toContainText("muonium, a real atom", { timeout: 30_000 });
  expect(errors).toEqual([]);
});
