/**
 * Light or dark, and how dense the plot grids are.
 *
 * Both are presentational and per viewer, so they live in localStorage and on
 * <html> as data attributes the stylesheet reads, not in the store or the URL:
 * a shared link should open in the reader's own theme. With no saved choice
 * the theme follows the OS (index.css pairs every colour with light-dark()).
 * index.html applies both saved choices before first paint, so there is no flash.
 */

export type Theme = "light" | "dark";
export type GridDensity = "fine" | "major" | "off";

const THEME_KEY = "atomsim-theme";
const GRID_KEY = "atomsim-grid";
export const GRID_ORDER: GridDensity[] = ["fine", "major", "off"];

function write(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Private windows and blocked storage: the choice just won't persist.
  }
}

/** The theme on screen right now: the saved choice, else the OS preference. */
export function currentTheme(): Theme {
  const set = document.documentElement.dataset.theme;
  if (set === "light" || set === "dark") return set;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function setTheme(t: Theme) {
  document.documentElement.dataset.theme = t;
  write(THEME_KEY, t);
}

export function currentGrid(): GridDensity {
  const g = document.documentElement.dataset.grid;
  return GRID_ORDER.includes(g as GridDensity) ? (g as GridDensity) : "fine";
}

export function setGrid(g: GridDensity) {
  document.documentElement.dataset.grid = g;
  write(GRID_KEY, g);
  window.dispatchEvent(new Event(GRID_KEY));
}

/** For useSyncExternalStore: every plot's grid button shows the same state. */
export function subscribeGrid(onChange: () => void): () => void {
  window.addEventListener(GRID_KEY, onChange);
  return () => window.removeEventListener(GRID_KEY, onChange);
}

/** The next density in the cycle the grid button walks. */
export function nextGrid(g: GridDensity): GridDensity {
  return GRID_ORDER[(GRID_ORDER.indexOf(g) + 1) % GRID_ORDER.length];
}

