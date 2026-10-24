/**
 * The sp, sp2 and sp3 hybrids the engine can draw (hybrids.py), and how to name them.
 *
 * Mirrored rather than fetched: the list is part of the URL contract (?hyb=sp3-1),
 * and a link has to be validated before anything has loaded.
 */
export const HYBRID_FAMILIES = [
  { family: "sp", lobes: 2 },
  { family: "sp2", lobes: 3 },
  { family: "sp3", lobes: 4 },
] as const;

export const HYBRID_KEYS: string[] = HYBRID_FAMILIES.flatMap(({ family, lobes }) =>
  Array.from({ length: lobes }, (_, i) => `${family}-${i + 1}`),
);

const SUP: Record<string, string> = { sp: "sp", sp2: "sp²", sp3: "sp³" };

/** "sp3-2" -> "sp³ hybrid, lobe 2 of 4". */
export function hybridLabel(key: string): string {
  const [family, idx] = key.split("-");
  const lobes = HYBRID_FAMILIES.find((f) => f.family === family)?.lobes ?? 0;
  return `${SUP[family] ?? family} hybrid, lobe ${idx} of ${lobes}`;
}

export function isHybridKey(key: string | null): key is string {
  return key !== null && HYBRID_KEYS.includes(key);
}
