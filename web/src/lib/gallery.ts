export interface StateRef {
  n: number;
  l: number;
  m: number;
}

/** Every (l, m) state of shell n, which is the row I lay out in the gallery. n² entries. */
export function galleryStates(n: number): StateRef[] {
  const out: StateRef[] = [];
  for (let l = 0; l < n; l++) {
    // I add 0 to normalize -0 (which -l gives me when l === 0) to +0
    for (let m = -l; m <= l; m++) out.push({ n, l, m: m + 0 });
  }
  return out;
}
