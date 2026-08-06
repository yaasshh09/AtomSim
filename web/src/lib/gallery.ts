export interface StateRef {
  n: number;
  l: number;
  m: number;
}

/** All (l, m) states of shell n, the gallery row. n² entries. */
export function galleryStates(n: number): StateRef[] {
  const out: StateRef[] = [];
  for (let l = 0; l < n; l++) {
    // + 0 normalizes -0 (from -l when l === 0) to +0
    for (let m = -l; m <= l; m++) out.push({ n, l, m: m + 0 });
  }
  return out;
}
