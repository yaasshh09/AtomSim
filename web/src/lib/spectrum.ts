const SERIES_NAMES = ["", "Lyman", "Balmer", "Paschen", "Brackett", "Pfund", "Humphreys"];
const SERIES_COLORS = ["", "#a78bfa", "#7cffb2", "#fbbf24", "#60a5fa", "#f472b6", "#f87171"];

export function seriesName(nLower: number): string {
  return SERIES_NAMES[nLower] ?? `to n'=${nLower}`;
}

export function seriesColor(nLower: number): string {
  return SERIES_COLORS[nLower] ?? "#8b98a5";
}
