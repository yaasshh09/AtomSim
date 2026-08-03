import type { HFLevels, ManyElectronParams, SystemInfo } from "../api/types";
import type { AtomModel } from "../state/store";

/**
 * The short form of the claim the provenance carries in full.
 *
 * It is a statement about the physics rather than about the rendering, so it
 * is not a disclosed liberty and does not flow through lib/liberties.ts. The
 * presentational disclosures already attached to the cloud and the surface are
 * unaffected and still apply.
 */
export const HF_ORBITAL_CAPTION =
  "One orbital of a self-consistent field, and an orbital is not an " +
  "observable. This atom's total density is exactly spherical, so the shape " +
  "here is a basis choice rather than a photograph. The lobes are still this " +
  "model's own answer: restricted Hartree-Fock leaves the angular part exactly Yₗₘ.";

/** The subset of app state a job payload needs to name its many-electron model. */
export interface ModelSelection {
  model: AtomModel;
  config: string | null;
  exchange: boolean;
  pauli: boolean;
}

/**
 * The four fields every picture job sends.
 *
 * The counterfactual flags are forced back to real physics under the screened
 * model. GSZ has no exchange term to remove and no occupancy cap of its own,
 * so a false here would be a flag the server accepts, does not honour, and
 * echoes into a badge over a picture that never departed from anything.
 */
export function manyElectronParams(s: ModelSelection): ManyElectronParams {
  return {
    model: s.model,
    config: s.config,
    exchange: s.model === "hf" ? s.exchange : true,
    pauli: s.model === "hf" ? s.pauli : true,
  };
}

/**
 * Whether the GSZ screened model can speak for this atom at all.
 *
 * True while the system table is still loading: the flag is a fact about an
 * atom the client has not been told about yet, and greying a control on a
 * guess is worse than greying it a moment late. Also true for every
 * hydrogenic preset, which no screened model touches.
 */
export function gszAvailable(systems: SystemInfo[], system: string): boolean {
  const info = systems.find((s) => s.key === system);
  return info === undefined || info.has_gsz;
}

/**
 * The model to actually use for this atom.
 *
 * Sulfur and chlorine have no GSZ parameters, so leaving "gsz" selected there
 * would mean every request refused with a 400 the picker could have avoided.
 * Applied both when the system changes and when the system table arrives,
 * because a deep link can name `?system=s&model=gsz` before the table that
 * knows better has loaded.
 */
export function resolveModel(
  systems: SystemInfo[],
  system: string,
  model: AtomModel,
): AtomModel {
  return gszAvailable(systems, system) ? model : "hf";
}

/**
 * Whether (n, l) can be drawn under the current model.
 *
 * Hartree-Fock builds one Fock operator per occupied subshell, so an empty one
 * has nothing to be an eigenfunction of and the server refuses it. Reading the
 * solve's own orbital list is what lets the picker grey the control instead of
 * firing a job it can already tell will come back 422.
 *
 * True when the solve has not landed yet: not knowing is not the same as
 * knowing the subshell is empty.
 */
export function subshellAvailable(
  hf: HFLevels | null,
  model: AtomModel,
  n: number,
  l: number,
): boolean {
  if (model !== "hf" || hf === null) return true;
  return hf.orbitals.some((o) => o.n === n && o.l === l);
}
