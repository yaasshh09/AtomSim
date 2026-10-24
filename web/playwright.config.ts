import { defineConfig } from "@playwright/test";

/**
 * End-to-end checks against the real server serving the real build.
 *
 * `npm run build` first: the server mounts web/dist. ATOMSIM_CMD names the
 * server command where `atomsim` is not on PATH (a conda env on Windows).
 * Files are *.e2e.ts so Vitest's default *.spec/*.test globs never pick them up.
 */
const PORT = 8123;

export default defineConfig({
  testDir: "e2e",
  testMatch: "**/*.e2e.ts",
  timeout: 90_000,
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: `http://127.0.0.1:${PORT}`, viewport: { width: 1400, height: 1000 } },
  webServer: {
    command: `${process.env.ATOMSIM_CMD ?? "atomsim"} serve --port ${PORT} --no-browser`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    // The suite would otherwise share one rate-limit bucket with itself.
    env: { ATOMSIM_RATE_LIMIT: "off" },
  },
});
