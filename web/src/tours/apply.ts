import type { SystemInfo } from "../api/types";
import { resolveModel } from "../lib/hfModel";
import type { UrlState } from "../lib/urlState";
import { INVALIDATED } from "../state/store";

/**
 * The store patch that puts the app into a tour step's state.
 *
 * A step can change n, l, m, system, config, model, exchange and pauli in one
 * move, so it must clear the union of what every individual action clears, not
 * just INVALIDATED. `INVALIDATED` is spread inside each action rather than
 * applied by `setState`, so a raw `setState(stepState(step))` would leave the
 * previous step's cloud, plane, surface, levels and spectrum in the store and
 * render them under this step's labels. `classicalGhost` (cleared by
 * setQuantumNumbers and setSystem), `hf` (setSystem, setConfig, setExchange,
 * setPauli) and `forceLaw` (setSystem) all live outside INVALIDATED, so they
 * are named here.
 *
 * INVALIDATED comes after the step's own state, which matters for the one
 * field they share: `profileZoom`. A zoom window names a line in a spectrum
 * this step has just discarded, so clearing it is the right way round.
 *
 * The model is resolved exactly as setSystem resolves it: sulfur and chlorine
 * have no GSZ parameters, and a step landing on one under the default model
 * would ask the server for a refusal on every request. The content test in
 * registry.test.ts requires such a step to name `model: "hf"` itself; this is
 * the belt to that pair of braces.
 *
 * `compare` is deliberately not resolved here. resolveCompare reads false
 * while the system table is still loading, which is exactly the state a tour
 * deep link opens in, and unlike `model` nothing re-resolves it when the table
 * lands. The radial request already applies that guard at the moment it can
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
