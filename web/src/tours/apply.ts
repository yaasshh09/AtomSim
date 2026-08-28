import type { SystemInfo } from "../api/types";
import { resolveModel } from "../lib/hfModel";
import type { UrlState } from "../lib/urlState";
import { INVALIDATED } from "../state/store";

/**
 * The store patch I use to put myself into a tour step's state.
 *
 * A step can change n, l, m, system, config, model, exchange and pauli in one
 * move, so I have to clear the union of what every individual action clears,
 * not just INVALIDATED. I spread `INVALIDATED` inside each action rather than
 * applying it in `setState`, so a raw `setState(stepState(step))` would leave
 * the previous step's cloud, plane, surface, levels and spectrum in my store
 * and render them under this step's labels. `classicalGhost` (which
 * setQuantumNumbers and setSystem clear), `hf` (setSystem, setConfig,
 * setExchange, setPauli) and `forceLaw` (setSystem) all live outside
 * INVALIDATED, so I name them here.
 *
 * I put INVALIDATED after the step's own state, which matters for the one
 * field they share: `profileZoom`. A zoom window names a line in a spectrum
 * this step has just discarded, so clearing it is the right way round.
 *
 * I resolve the model exactly as setSystem resolves it: sulfur and chlorine
 * have no GSZ parameters, and a step landing on one under my default model
 * would ask my server for a refusal on every request. My content test in
 * registry.test.ts requires such a step to name `model: "hf"` itself; this is
 * the belt to that pair of braces.
 *
 * I deliberately do not resolve `compare` here. resolveCompare reads false
 * while my system table is still loading, which is exactly the state a tour
 * deep link opens in, and unlike `model` nothing re-resolves it when the table
 * lands. My radial request already applies that guard at the moment it can
 * actually be answered, so forcing it here would silently cost a step the
 * comparison it exists to show.
 */
export function tourReset(state: UrlState, systems: SystemInfo[]) {
  return {
    ...state,
    model: resolveModel(systems, state.system, state.model),
    ...INVALIDATED,
    classicalGhost: null,
    classicalStatus: "idle" as const,
    hf: null,
    hfStatus: "idle" as const,
    forceLaw: null,
    forceStatus: "idle" as const,
  };
}
