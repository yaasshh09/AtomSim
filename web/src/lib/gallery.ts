export interface StateRef {
  n: number;
  l: number;
  m: number;
}

/** Every (l, m) state of shell n, which is the row the gallery lays out. n² entries. */
export function galleryStates(n: number): StateRef[] {
  const out: StateRef[] = [];
  for (let l = 0; l < n; l++) {
    // Adding 0 normalizes -0 (which -l gives when l === 0) to +0
    for (let m = -l; m <= l; m++) out.push({ n, l, m: m + 0 });
  }
  return out;
}
