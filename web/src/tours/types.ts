import type { UrlState } from "../lib/urlState";

/**
 * The quantities I let a tour step assert about my engine.
 *
 * I keep this deliberately narrow. A resolver that silently returned the wrong
 * quantity would put a green tick on prose that lies, which is worse than
 * having no test at all, so each kind earns its place with its own test
 * against a value known in closed form. Adding a kind is one function in my
 * Python dispatch table plus one entry here; my structural test asserts the
 * two lists match.
 */
export const CLAIM_KINDS = ["energy_eV", "mean_r_pm", "wavelength_nm", "ionization_eV"] as const;

export type ClaimKind = (typeof CLAIM_KINDS)[number];

/**
 * One numeric assertion a step's prose makes.
 *
 * It carries its own inputs and inherits the step's `state` for anything it
 * does not name, so I can check a claim standing alone. `tol` is absolute, in
 * the claim's own unit, and I require it: a tolerance the author had to choose
 * is a tolerance the author had to think about.
 */
export interface Claim {
  of: ClaimKind;
  is: number;
  tol: number;
  system?: string;
  n?: number;
  l?: number;
  model?: "gsz" | "hf";
  fineStructure?: boolean;
  dirac?: boolean;
  exchange?: boolean;
  pauli?: boolean;
  n_upper?: number;
  n_lower?: number;
}

export interface TourStep {
  id: string;
  title: string;
  /** Paragraphs. I render one <p> each and interpret no markup. */
  body: string[];
  /** A partial UrlState. I apply it over URL_DEFAULTS, never over the previous step. */
  state: Partial<UrlState>;
  /** A `data-tour` anchor for me to ring, or absent when I should draw no ring. */
  spotlight?: string;
  claims?: Claim[];
}

export interface Tour {
  id: string;
  title: string;
  blurb: string;
  steps: TourStep[];
}
