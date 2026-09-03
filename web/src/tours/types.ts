import type { UrlState } from "../lib/urlState";

/**
 * The quantities a tour step may assert about the engine.
 *
 * Deliberately narrow. A resolver that silently returned the wrong quantity
 * would put a green tick on prose that lies, which is worse than having no
 * test at all, so each kind earns its place with its own test against a value
 * known in closed form. Adding a kind is one function in the Python dispatch
 * table plus one entry here, and the structural test asserts the two lists
 * match.
 */
export const CLAIM_KINDS = ["energy_eV", "mean_r_pm", "wavelength_nm", "ionization_eV"] as const;

export type ClaimKind = (typeof CLAIM_KINDS)[number];

/**
 * One numeric assertion a step's prose makes.
 *
 * It carries its own inputs and inherits the step's `state` for anything it
 * does not name, so a claim can be checked standing alone. `tol` is absolute,
 * in the claim's own unit, and required: a tolerance the author had to choose
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
  /** Paragraphs. One <p> each, with no markup interpreted. */
  body: string[];
  /** A partial UrlState, applied over URL_DEFAULTS, never over the previous step. */
  state: Partial<UrlState>;
  /** A `data-tour` anchor to ring, or absent when no ring should be drawn. */
  spotlight?: string;
  claims?: Claim[];
}

export interface Tour {
  id: string;
  title: string;
  blurb: string;
  steps: TourStep[];
}
