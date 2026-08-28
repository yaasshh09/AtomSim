import type { HFLevels, ManyElectronParams, SystemInfo } from "../api/types";
import type { AtomModel } from "../state/store";

/**
 * The short form of the claim my provenance carries in full.
 *
 * It is a statement about the physics rather than about my rendering, so it is
 * not a disclosed liberty and I do not send it through lib/liberties.ts. The
 * presentational disclosures I already attach to the cloud and the surface are
 * unaffected and still apply.
 */
export const HF_ORBITAL_CAPTION =
  "I am drawing one orbital of a self-consistent field, and an orbital is not " +
  "an observable. This atom's total density is exactly spherical, so the shape " +
  "you see is a basis choice rather than a photograph. The lobes are still " +
  "this model's own answer: restricted Hartree-Fock leaves the angular part " +
  "exactly Yₗₘ.";

/** The part of my state a job payload needs in order to name its many-electron model. */
export interface ModelSelection {
  model: AtomModel;
  config: string | null;
  exchange: boolean;
  pauli: boolean;
}

/**
 * The four fields I send with every picture job.
 *
 * I force the counterfactual flags back to real physics under the screened
 * model. GSZ has no exchange term for me to remove and no occupancy cap of its
 * own, so a false here would be a flag my server accepts, does not honour, and
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
 * I answer true while my system table is still loading: the flag is a fact
 * about an atom nobody has told me about yet, and greying a control on a guess
 * is worse than greying it a moment late. I also answer true for every
 * hydrogenic preset, which no screened model touches.
 */
export function gszAvailable(systems: SystemInfo[], system: string): boolean {
  const info = systems.find((s) => s.key === system);
  return info === undefined || info.has_gsz;
}

/**
 * The model I actually use for this atom.
 *
 * Sulfur and chlorine have no GSZ parameters, so if I left "gsz" selected
 * there I would have every request refused with a 400 my picker could have
 * avoided. I apply this both when the system changes and when my system table
 * arrives, because a deep link can name `?system=s&model=gsz` before the table
 * that knows better has loaded.
 */
export function resolveModel(
  systems: SystemInfo[],
  system: string,
  model: AtomModel,
): AtomModel {
  return gszAvailable(systems, system) ? model : "hf";
}

/**
 * Whether both of my models can draw this atom's total density.
 *
 * I need a many-electron atom (a one-electron system has no total density for
 * either model) and GSZ parameters (sulfur and chlorine have none). I answer
 * false while my system table is still loading, which is the opposite default
 * from `gszAvailable` and deliberate: that one greys a control, and greying
 * late is harmless, while this one gates a request that would come back 422 if
 * a deep link named it before the table landed.
 */
export function compareAvailable(systems: SystemInfo[], system: string): boolean {
  const info = systems.find((s) => s.key === system);
  return info !== undefined && info.kind === "screened" && info.has_gsz;
}

/** I force a deep-linked `compare=1` off where I cannot run it. */
export function resolveCompare(
  systems: SystemInfo[],
  system: string,
  compare: boolean,
): boolean {
  return compareAvailable(systems, system) && compare;
}

/**
 * Whether I can draw (n, l) under the current model.
 *
 * Hartree-Fock builds one Fock operator per occupied subshell, so an empty one
 * has nothing to be an eigenfunction of and my server refuses it. Reading the
 * solve's own orbital list is what lets my picker grey the control instead of
 * firing a job I can already tell will come back 422.
 *
 * I answer true when the solve has not landed yet: not knowing is not the same
 * as knowing the subshell is empty.
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
