/** The transitions the laser panel offers, all driven with z-polarized (pi) light at m = 0. */
export const RABI_TRANSITIONS = [
  { value: "1s-2p", n1: 1, l1: 0, n2: 2, l2: 1, hint: "Lyman-α, 121.6 nm. 2p can only fall back to 1s, so the pair is closed." },
  { value: "1s-3p", n1: 1, l1: 0, n2: 3, l2: 1, hint: "Lyman-β, 102.6 nm. 3p sometimes falls to 2s instead and is lost to the laser." },
  { value: "2s-3p", n1: 2, l1: 0, n2: 3, l2: 1, hint: "Hα from the metastable 2s, 656 nm. 3p mostly falls to 1s and never comes back." },
  { value: "2p-3d", n1: 2, l1: 1, n2: 3, l2: 2, hint: "Hα, 656 nm. 3d returns to 2p, but often to a sublevel this laser does not touch." },
] as const;

export type RabiTransition = (typeof RABI_TRANSITIONS)[number]["value"];

const PREFIXES: [number, string][] = [
  [1e12, "T"],
  [1e9, "G"],
  [1e6, "M"],
  [1e3, "k"],
  [1, ""],
];

/** An intensity in W/m² with an SI prefix: 3.2e7 -> "32 MW/m²". */
export function formatIntensity(wm2: number): string {
  const [scale, p] = PREFIXES.find(([f]) => wm2 >= f) ?? PREFIXES[PREFIXES.length - 1];
  return `${Number((wm2 / scale).toPrecision(2))} ${p}W/m²`;
}
