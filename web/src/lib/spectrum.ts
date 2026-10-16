const SERIES_NAMES = ["", "Lyman", "Balmer", "Paschen", "Brackett", "Pfund", "Humphreys"];
// Mid-luminance hues, so each series reads on the white theme and the black one.
const SERIES_COLORS = ["", "#8b5cf6", "#22a35a", "#d4900a", "#3b82f6", "#e0479e", "#e5484d"];

export function seriesName(nLower: number): string {
  return SERIES_NAMES[nLower] ?? `to n'=${nLower}`;
}

export function seriesColor(nLower: number): string {
  return SERIES_COLORS[nLower] ?? "#8b98a5";
}
