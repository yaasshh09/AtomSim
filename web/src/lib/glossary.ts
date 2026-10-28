/**
 * The words this app uses that a newcomer may not know, each in a sentence or
 * two of plain language. Keys are matched case-insensitively against running
 * text, longest first, so "Hartree-Fock" wins over "hartree".
 */
export const GLOSSARY: Record<string, string> = {
  orbital:
    "A one-electron wavefunction: where one electron is likely to be found, and with what phase.",
  wavefunction:
    "The quantity ψ that quantum mechanics evolves. Its squared size |ψ|² is a probability density.",
  "probability density":
    "|ψ|²: the chance per unit volume of finding the electron at a point.",
  "radial function":
    "The part of a hydrogen-like wavefunction that depends only on the distance r from the nucleus.",
  node: "A place where the wavefunction is exactly zero, so the electron is never found there.",
  "quantum number":
    "An integer that labels a state: n sets the energy shell, ℓ the angular momentum, m its orientation.",
  hartree:
    "The atomic unit of energy, about 27.2 eV: twice the energy needed to ionize hydrogen from its ground state.",
  bohr: "The atomic unit of length, about 52.9 pm: the radius of the Bohr model's smallest orbit.",
  "reduced mass":
    "The effective mass in a two-body problem, mM/(m + M). It is why deuterium's lines sit slightly off hydrogen's.",
  "fine structure":
    "Small splittings of hydrogen's levels from relativity and the electron's spin, about α² of the level spacing.",
  "Lamb shift":
    "A tiny splitting of levels that the Dirac equation says should coincide, caused by quantum electrodynamics.",
  hyperfine:
    "Splittings from the magnetic interaction of the electron with the nucleus. Hydrogen's gives the 21 cm radio line.",
  "Zeeman effect": "The splitting of spectral lines in a magnetic field.",
  "Stark effect": "The splitting and shifting of spectral lines in an electric field.",
  "Hartree-Fock":
    "A self-consistent method for many-electron atoms: each electron moves in the average field of the others, with exchange included.",
  "correlation energy":
    "The energy Hartree-Fock misses because it lets electrons avoid each other only on average, not instant by instant.",
  exchange:
    "A purely quantum effect of identical electrons: electrons of the same spin keep apart, which lowers the energy.",
  "Pauli exclusion principle":
    "No two electrons in an atom can share all their quantum numbers, so they fill shells instead of all crowding into 1s.",
  screening:
    "Inner electrons partly cancel the nucleus's charge, so outer electrons feel a smaller effective charge.",
  "ionization energy": "The energy needed to remove an electron from an atom or ion.",
  "Koopmans' theorem":
    "Reads an ionization energy off a Hartree-Fock solve as minus an orbital's energy, assuming the other electrons do not rearrange.",
  "quantum defect":
    "How far an alkali's highly excited levels sit below hydrogen's, written as a shift of n.",
  "isoelectronic sequence":
    "Atoms and ions with the same number of electrons but different nuclear charge, such as He, Li⁺ and Be²⁺.",
  superposition:
    "A state that is a sum of several stationary states. Its density moves, at the frequencies of the energy differences.",
  "stationary state":
    "A state of definite energy. Its probability density never changes in time.",
  "Einstein A coefficient": "The rate at which an excited state decays by emitting light.",
  "oscillator strength":
    "A dimensionless measure of how strongly a transition absorbs or emits light.",
  "Rabi oscillation":
    "The back-and-forth transfer of an atom between two levels when a laser drives it on resonance.",
  isosurface: "A surface on which a quantity takes one fixed value: here, a contour of |ψ|².",
  "Moseley's law":
    "The square root of an element's X-ray line energy rises in a straight line with its atomic number.",
  counterfactual:
    "A deliberately altered physics, solved rigorously, to show what a law or constant is responsible for.",
};

export interface GlossPart {
  text: string;
  /** The glossary key this span matches, or undefined for ordinary text. */
  term?: string;
}

const KEYS = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length);

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const PATTERN = new RegExp(
  `(?<![\\p{L}\\p{N}])(${KEYS.map(escapeRegex).join("|")})(?![\\p{L}\\p{N}])`,
  "giu",
);

/**
 * Splits text into plain runs and glossary terms, marking only the first
 * occurrence of each term, so a paragraph gets one hover per idea rather than
 * an underline on every repeat.
 */
export function glossParts(text: string): GlossPart[] {
  const out: GlossPart[] = [];
  const seen = new Set<string>();
  let last = 0;
  for (const match of text.matchAll(PATTERN)) {
    const key = KEYS.find((k) => k.toLowerCase() === match[0].toLowerCase());
    if (!key || seen.has(key)) continue;
    seen.add(key);
    const at = match.index ?? 0;
    if (at > last) out.push({ text: text.slice(last, at) });
    out.push({ text: match[0], term: key });
    last = at + match[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}
