import type { Provenance } from "../api/types";

/**
 * BibTeX for a value: the software that computed it, plus whatever primary
 * source its provenance names.
 *
 * Only sources this repository already cites word for word are offered here
 * (the screening paper in numerics/screening.py, the NIST ASD release in the
 * vendored data, CODATA as scipy distributes it). A citation guessed from
 * memory is a fabricated record, which is the one thing this file must never
 * produce, so a source that is not on this list is simply not cited.
 */
const SOURCES: { key: string; match: RegExp; bibtex: string }[] = [
  {
    key: "szydlik1974",
    match: /\bGSZ\b|Szydlik|screened/i,
    bibtex: `@article{szydlik1974,
  author  = {Szydlik, P. P. and Green, A. E. S.},
  title   = {Independent-particle-model potentials for ions and neutral atoms with {Z} <= 18},
  journal = {Physical Review A},
  volume  = {9},
  pages   = {1885},
  year    = {1974}
}`,
  },
  {
    key: "nistasd",
    match: /\bNIST\b/,
    bibtex: `@misc{nistasd,
  author       = {Kramida, A. and Ralchenko, Yu. and Reader, J. and {NIST ASD Team}},
  title        = {{NIST} Atomic Spectra Database (ver. 5.12)},
  howpublished = {\\url{https://physics.nist.gov/asd}}
}`,
  },
  {
    key: "codata",
    match: /CODATA/,
    bibtex: `@misc{codata,
  title        = {{CODATA} recommended values of the fundamental physical constants},
  note         = {as distributed in scipy.constants},
  howpublished = {\\url{https://physics.nist.gov/constants}}
}`,
  },
];

export function atomsimEntry(version: string, today: string): string {
  return `@software{atomsim,
  author  = {Gupta, Yash},
  title   = {{AtomSim}: an atomic-physics engine that never quietly lies},
  version = {${version}},
  url     = {https://github.com/yaasshh09/AtomSim},
  urldate = {${today}},
  note    = {CC-BY-NC-4.0}
}`;
}

/** The BibTeX block for one provenance: a comment saying what is cited, then the entries. */
export function citeBibtex(p: Provenance, version: string, today: string): string {
  const text = [p.method, ...p.assumptions].join(" ");
  const extra = SOURCES.filter((s) => s.match.test(text)).map((s) => s.bibtex);
  const comment = `% ${p.fidelity.toUpperCase()}: ${p.method}`.replace(/\n/g, " ");
  return [comment, atomsimEntry(version, today), ...extra].join("\n\n");
}
