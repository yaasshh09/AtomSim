import type { UrlState } from "../lib/urlState";

/**
 * The quantities a tour step may assert about the engine.
 *
 * Deliberately narrow. A resolver that silently returns the wrong quantity
 * reports a green tick on prose that lies, which is worse than having no test
 * at all, so each kind earns its place with its own test against a value known
 * in closed form. Adding a kind is one function in the Python dispatch table
 * plus one entry here; the structural test asserts the two lists match.
 */
export const CLAIM_KINDS = ["energy_eV", "mean_r_pm", "wavelength_nm", "ionization_eV"] as const;

export type ClaimKind = (typeof CLAIM_KINDS)[number];

/**
 * One numeric assertion a step's prose makes.
 *
 * Carries its own inputs and inherits the step's `state` for anything it does
 * not name, so a claim is checkable standing alone. `tol` is absolute, in the
 * claim's own unit, and is required: a tolerance the author had to choose is a
 * tolerance the author had to think about.
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
  /** Paragraphs. Rendered one <p> each; no markup is interpreted. */
  body: string[];
  /** Partial UrlState. Applied over URL_DEFAULTS, never over the previous step. */
  state: Partial<UrlState>;
  /** A `data-tour` anchor to ring, or absent for no ring. */
  spotlight?: string;
  claims?: Claim[];
}

export interface Tour {
  id: string;
  title: string;
  blurb: string;
  steps: TourStep[];
}
