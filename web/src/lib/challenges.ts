import type { SystemInfo } from "../api/types";
import type { UrlState } from "./urlState";

/**
 * Small puzzles answered by putting the app into the right state.
 *
 * Every check reads only the URL state (and the systems table where a question
 * is about a system's size), so a challenge is solved by what is on screen and
 * nothing else, and a solution can be written down as a link. Each one carries
 * the reason its answer is the answer, shown once it is solved.
 */
export interface Challenge {
  id: string;
  prompt: string;
  hint: string;
  why: string;
  check: (s: UrlState, systems: SystemInfo[]) => boolean;
  /** One state that solves it, applied over the defaults; the tests hold the check to it. */
  solution: Partial<UrlState>;
}

const sizeFactor = (systems: SystemInfo[], key: string): number | null => {
  const s = systems.find((x) => x.key === key);
  return s ? s.z * s.mu_ratio.value : null;
};

export const CHALLENGES: Challenge[] = [
  {
    id: "two-radial-one-angular",
    prompt: "Find a state with exactly two radial nodes and one angular node.",
    hint: "Radial nodes number n − ℓ − 1; angular nodes number ℓ.",
    why: "ℓ = 1 gives one angular node, and n − ℓ − 1 = 2 then needs n = 4: any 4p state.",
    check: (s) => s.l === 1 && s.n - s.l - 1 === 2,
    solution: { n: 4, l: 1, m: 0 },
  },
  {
    id: "nodal-cone",
    prompt: "Find a hydrogen state whose density has a nodal cone around the z axis.",
    hint: "The polar nodes come from the associated Legendre function: there are ℓ − |m| of them.",
    why: "With ℓ − |m| of at least 2, at least one polar node sits away from the xy plane, and a node at fixed θ is a cone.",
    check: (s) => s.l - Math.abs(s.m) >= 2,
    solution: { n: 3, l: 2, m: 0 },
  },
  {
    id: "smaller-than-hydrogen",
    prompt: "Make the 1s orbital smaller than hydrogen's by changing only the system.",
    hint: "Sizes scale as 1/(Z μ).",
    why: "A larger nuclear charge or a heavier orbiting particle pulls the orbital in: He⁺ is half the size, muonic hydrogen about 186 times smaller.",
    check: (s, systems) => {
      const here = sizeFactor(systems, s.system);
      const h = sizeFactor(systems, "h");
      return s.n === 1 && here !== null && h !== null && here > h * (1 + 1e-9);
    },
    solution: { n: 1, l: 0, m: 0, system: "he+" },
  },
  {
    id: "lamb-shift",
    prompt: "Show two hydrogen levels that the Dirac equation says should coincide.",
    hint: "Look at the level ladder with the relativistic detail switched on.",
    why: "Dirac's levels depend only on n and j, so 2s½ and 2p½ coincide exactly; the measured gap between them is the Lamb shift, from QED.",
    check: (s) => s.view === "levels" && s.dirac,
    solution: { view: "levels", dirac: true },
  },
  {
    id: "radio-line",
    prompt: "Find the splitting behind hydrogen's 21 cm radio line.",
    hint: "It comes from the nucleus, not from the electron alone.",
    why: "The electron's and proton's spins can be parallel or antiparallel; the energy difference is about 5.9 μeV, a photon of 21 cm.",
    check: (s) => s.view === "levels" && s.hyperfine,
    solution: { view: "levels", hyperfine: true },
  },
  {
    id: "paschen-back",
    prompt: "Apply a magnetic field strong enough to swamp the fine structure.",
    hint: "The fine-structure gap in n = 2 is about 45 μeV, and the Zeeman shift is about 58 μeV per tesla.",
    why: "Past a few tesla the field wins: spin and orbit decouple and the levels fan out by mℓ + 2mₛ, the Paschen-Back regime.",
    check: (s) => s.view === "levels" && s.bField >= 5,
    solution: { view: "levels", bField: 5 },
  },
  {
    id: "no-pauli",
    prompt: "Build an atom without the exclusion principle.",
    hint: "It is a counterfactual switch on the Hartree-Fock model.",
    why: "With no occupancy cap every electron falls into 1s, so there are no shells, and with no shells there is no chemistry.",
    check: (s) => s.model === "hf" && !s.pauli,
    solution: { system: "ne", model: "hf", exchange: false, pauli: false, view: "levels" },
  },
  {
    id: "vanishing-slice",
    prompt: "Tilt the 2-D slice so that a 2p₀ orbital disappears completely.",
    hint: "2p₀ is odd in z.",
    why: "2p₀ goes as z, so it is exactly zero on the plane z = 0: lay the slice flat through the nucleus.",
    check: (s) =>
      s.view === "plane" &&
      s.n === 2 &&
      s.l === 1 &&
      s.m === 0 &&
      s.planeSlice.theta === 0 &&
      s.planeSlice.offset === 0,
    solution: {
      n: 2,
      l: 1,
      m: 0,
      view: "plane",
      planeSlice: { theta: 0, phi: 0, offset: 0 },
    },
  },
  {
    id: "correlation",
    prompt: "Find out how much energy Hartree-Fock misses in helium.",
    hint: "Open helium's Hartree-Fock level ladder and look below it.",
    why: "About 1.14 eV: the correlation energy, the price of letting each electron see the other only as an average cloud.",
    check: (s) => s.system === "he" && s.model === "hf" && s.view === "levels",
    solution: { system: "he", model: "hf", view: "levels" },
  },
];

/** The challenges this state solves, by id. */
export function solvedBy(s: UrlState, systems: SystemInfo[]): Set<string> {
  return new Set(CHALLENGES.filter((c) => c.check(s, systems)).map((c) => c.id));
}
