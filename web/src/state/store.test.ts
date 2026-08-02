import { beforeEach, describe, expect, it } from "vitest";
import { defaultParams } from "../lib/forceLaw";
import { useAppStore } from "./store";

const initial = useAppStore.getState();

beforeEach(() => {
  useAppStore.setState(initial, true);
});

function pretendLoaded() {
  useAppStore.setState({
    positions: new Float32Array(3),
    density: new Float32Array(1),
    phase: new Float32Array(1),
    stateInfo: {} as never,
    plane: {} as never,
    radial: {} as never,
    levels: {} as never,
    spectrum: {} as never,
    status: "ready",
  });
}

describe("store transitions", () => {
  it("clamps quantum numbers and invalidates data", () => {
    pretendLoaded();
    useAppStore.getState().setQuantumNumbers(3, 5, -9);
    const s = useAppStore.getState();
    expect([s.n, s.l, s.m]).toEqual([3, 2, -2]);
    expect(s.positions).toBeNull();
    expect(s.plane).toBeNull();
    expect(s.radial).toBeNull();
    expect(s.status).toBe("idle");
  });

  it("system change invalidates data", () => {
    pretendLoaded();
    useAppStore.getState().setSystem("mu-h");
    const s = useAppStore.getState();
    expect(s.system).toBe("mu-h");
    expect(s.stateInfo).toBeNull();
    expect(s.spectrum).toBeNull();
  });

  it("real basis demotes phase color mode", () => {
    useAppStore.setState({ colorMode: "phase" });
    useAppStore.getState().setBasis("real");
    const s = useAppStore.getState();
    expect(s.basis).toBe("real");
    expect(s.colorMode).toBe("density");
  });

  it("complex basis keeps chosen color mode", () => {
    useAppStore.setState({ colorMode: "density" });
    useAppStore.getState().setBasis("complex");
    expect(useAppStore.getState().colorMode).toBe("density");
  });

  it("fine-structure toggle clears only energy-derived data", () => {
    pretendLoaded();
    useAppStore.getState().setFineStructure(true);
    const s = useAppStore.getState();
    expect(s.fineStructure).toBe(true);
    expect(s.stateInfo).toBeNull();
    expect(s.levels).toBeNull();
    expect(s.spectrum).toBeNull();
    expect(s.positions).not.toBeNull();
  });

  it("plane quantity toggle clears only the plane", () => {
    pretendLoaded();
    useAppStore.getState().setPlaneQuantity("psi");
    const s = useAppStore.getState();
    expect(s.planeQuantity).toBe("psi");
    expect(s.plane).toBeNull();
    expect(s.positions).not.toBeNull();
  });

  it("lab constant change clears only the what-if data, not main physics", () => {
    pretendLoaded();
    useAppStore.setState({ whatif: {} as never, whatifStatus: "ready" });
    useAppStore.getState().setLabConst({ e: 2 });
    const s = useAppStore.getState();
    expect(s.labConst.e).toBe(2);
    expect(s.labConst.hbar).toBe(1);
    expect(s.whatif).toBeNull();
    expect(s.whatifStatus).toBe("idle");
    expect(s.positions).not.toBeNull();
    expect(s.levels).not.toBeNull();
  });

  it("lab Z change clears only the what-if data", () => {
    pretendLoaded();
    useAppStore.setState({ whatif: {} as never, whatifStatus: "ready" });
    useAppStore.getState().setLabZ(3);
    const s = useAppStore.getState();
    expect(s.labZ).toBe(3);
    expect(s.whatif).toBeNull();
    expect(s.positions).not.toBeNull();
  });

  it("ghost toggle is off by default and flips without touching physics fields", async () => {
    pretendLoaded();
    const before = useAppStore.getState().positions;
    expect(useAppStore.getState().ghost).toBe(false);
    useAppStore.getState().setGhost(true);
    expect(useAppStore.getState().ghost).toBe(true);
    expect(useAppStore.getState().positions).toBe(before);
    // setGhost fired loadClassical (status was "idle"); let its fetch rejection
    // settle inside this test so the caught-error set() cannot leak into the next.
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  it("changing n or system clears loaded classical data (no stale ghost)", () => {
    useAppStore.setState({ classicalGhost: { n: 1 } as never, classicalStatus: "ready" });
    useAppStore.getState().setQuantumNumbers(2, 0, 0);
    expect(useAppStore.getState().classicalGhost).toBeNull();
    expect(useAppStore.getState().classicalStatus).toBe("idle");
  });

  it("changing system clears loaded classical data", () => {
    useAppStore.setState({ classicalGhost: { n: 1 } as never, classicalStatus: "ready" });
    useAppStore.getState().setSystem("he+");
    expect(useAppStore.getState().classicalGhost).toBeNull();
    expect(useAppStore.getState().classicalStatus).toBe("idle");
  });

  it("nucleus mode is a pure render choice: defaults to marker, clears nothing", () => {
    expect(useAppStore.getState().nucleusMode).toBe("marker");
    pretendLoaded();
    useAppStore.getState().setNucleusMode("true-scale");
    const s = useAppStore.getState();
    expect(s.nucleusMode).toBe("true-scale");
    expect(s.positions).not.toBeNull();
    expect(s.stateInfo).not.toBeNull();
  });
});

describe("force-law slice", () => {
  it("changing a param or l clears stale force-law data", () => {
    useAppStore.setState({ forceLaw: { preset: "powerlaw" } as never, forceStatus: "ready" });
    useAppStore.getState().setForceParam("p", 1.2);
    expect(useAppStore.getState().forceLaw).toBeNull();
    expect(useAppStore.getState().forceStatus).toBe("idle");

    useAppStore.setState({ forceLaw: { preset: "powerlaw" } as never, forceStatus: "ready" });
    useAppStore.getState().setForceL(1);
    expect(useAppStore.getState().forceLaw).toBeNull();
  });

  it("setForcePreset swaps params to that preset's defaults and clears data", () => {
    const s = useAppStore.getState();
    s.setForcePreset("yukawa");
    const st = useAppStore.getState();
    expect(st.forcePreset).toBe("yukawa");
    expect(st.forceParams).toEqual(defaultParams("yukawa"));
    expect(st.forceLaw).toBeNull();
    expect(st.forceStatus).toBe("idle");
  });

  it("setForceParam clamps and clears force-law data", () => {
    useAppStore.getState().setForcePreset("yukawa");
    useAppStore.getState().setForceParam("lambda", 999);
    expect(useAppStore.getState().forceParams.lambda).toBe(20); // spec max
    expect(useAppStore.getState().forceLaw).toBeNull();
  });

  it("setForceViz is presentational: it does not clear force-law data", () => {
    useAppStore.setState({ forceLaw: { preset: "powerlaw" } as never, forceStatus: "ready" });
    useAppStore.getState().setForceViz("ladder");
    expect(useAppStore.getState().forceViz).toBe("ladder");
    expect(useAppStore.getState().forceLaw).not.toBeNull(); // untouched
  });

  it("setSystem resets config to the Aufbau default (null) and clears physics", () => {
    useAppStore.getState().setConfig("1s2 2s1");
    useAppStore.getState().setSystem("na");
    const st = useAppStore.getState();
    expect(st.system).toBe("na");
    expect(st.config).toBeNull();
    expect(st.levels).toBeNull();
  });

  it("setConfig clears derived physics but keeps the system", () => {
    useAppStore.setState({ system: "na", levels: {} as never });
    useAppStore.getState().setConfig("1s2 2s2 2p6 3p1");
    expect(useAppStore.getState().config).toBe("1s2 2s2 2p6 3p1");
    expect(useAppStore.getState().levels).toBeNull();
  });

  it("setBField clears cached levels", () => {
    useAppStore.setState({ levels: { fake: true } as never, bField: 0 });
    useAppStore.getState().setBField(4);
    expect(useAppStore.getState().bField).toBe(4);
    expect(useAppStore.getState().levels).toBeNull();
  });

  it("setEField clears cached levels", () => {
    useAppStore.setState({ levels: { fake: true } as never, eField: 0 });
    useAppStore.getState().setEField(25);
    expect(useAppStore.getState().eField).toBe(25);
    expect(useAppStore.getState().levels).toBeNull();
  });

  it("setHyperfine toggles the flag and clears cached levels", () => {
    useAppStore.setState({ levels: { fake: true } as never, hyperfine: false });
    useAppStore.getState().setHyperfine(true);
    expect(useAppStore.getState().hyperfine).toBe(true);
    expect(useAppStore.getState().levels).toBeNull();
  });

  it("intensities default on, so the spectrum is never silently uniform", () => {
    expect(useAppStore.getInitialState().intensities).toBe(true);
  });

  it("setIntensities clears the cached spectrum, which was fetched without rates", () => {
    useAppStore.setState({ spectrum: { fake: true } as never, intensities: true });
    useAppStore.getState().setIntensities(false);
    expect(useAppStore.getState().intensities).toBe(false);
    expect(useAppStore.getState().spectrum).toBeNull();
  });
});

describe("many-electron model selection", () => {
  it("defaults to the screened model, so no existing session changes physics", () => {
    expect(useAppStore.getInitialState().model).toBe("gsz");
    expect(useAppStore.getInitialState().hf).toBeNull();
  });

  it("setModel invalidates everything derived under the previous model", () => {
    pretendLoaded();
    useAppStore.getState().setModel("hf");
    const s = useAppStore.getState();
    expect(s.model).toBe("hf");
    expect(s.levels).toBeNull();
    expect(s.stateInfo).toBeNull();
    expect(s.radial).toBeNull();
    expect(s.spectrum).toBeNull();
    expect(s.positions).toBeNull();
    expect(s.plane).toBeNull();
    expect(s.status).toBe("idle");
  });

  // The solve costs seconds and is keyed on the atom, not on which model is on
  // screen, so switching away and back must not throw it away.
  it("setModel keeps the solve itself: it is not derived from the model", () => {
    useAppStore.setState({ hf: { fake: true } as never, hfStatus: "ready" });
    useAppStore.getState().setModel("hf");
    expect(useAppStore.getState().hf).not.toBeNull();
    useAppStore.getState().setModel("gsz");
    expect(useAppStore.getState().hf).not.toBeNull();
  });

  // ...but it belongs to one atom in one configuration, and both of those are
  // physics inputs to the solve.
  it("setSystem drops the solve, which was for a different element", () => {
    useAppStore.setState({ hf: { fake: true } as never, hfStatus: "ready" });
    useAppStore.getState().setSystem("ar");
    expect(useAppStore.getState().hf).toBeNull();
    expect(useAppStore.getState().hfStatus).toBe("idle");
  });

  it("setConfig drops the solve, which was for a different configuration", () => {
    useAppStore.setState({
      system: "na", hf: { fake: true } as never, hfStatus: "ready",
    });
    useAppStore.getState().setConfig("1s2 2s2 2p6 3p1");
    expect(useAppStore.getState().hf).toBeNull();
    expect(useAppStore.getState().hfStatus).toBe("idle");
  });

  // An HF solve depends on the occupied subshells and on nothing else. Putting
  // it in INVALIDATED would be the safe-looking choice and would re-solve on
  // every click of n, which is seconds of wall time for no change in answer.
  it("changing n keeps the solve, which does not depend on the drawn state", () => {
    useAppStore.setState({ hf: { fake: true } as never, hfStatus: "ready" });
    useAppStore.getState().setQuantumNumbers(3, 1, 0);
    expect(useAppStore.getState().hf).not.toBeNull();
    expect(useAppStore.getState().hfStatus).toBe("ready");
  });
});

describe("exchange toggle (distinguishable electrons)", () => {
  it("defaults to real physics, so no session lands in the counterfactual", () => {
    expect(useAppStore.getInitialState().exchange).toBe(true);
  });

  // The two models are different atoms. A stale solve sitting under a flipped
  // switch would be a ladder labelled Hartree drawn from Hartree-Fock numbers,
  // which is exactly the quiet lie the badge exists to prevent.
  it("setExchange drops the solve, which was for the other model", () => {
    useAppStore.setState({ hf: { fake: true } as never, hfStatus: "ready" });
    useAppStore.getState().setExchange(false);
    expect(useAppStore.getState().exchange).toBe(false);
    expect(useAppStore.getState().hf).toBeNull();
    expect(useAppStore.getState().hfStatus).toBe("idle");
  });

  // "idle" and not "sampling": nothing has been requested yet, and a view that
  // said "solving" before a request existed would describe work nobody started.
  it("goes back to idle rather than pretending a solve is running", () => {
    useAppStore.setState({ hfStatus: "ready" });
    useAppStore.getState().setExchange(false);
    expect(useAppStore.getState().hfStatus).toBe("idle");
  });

  it("altered physics does not follow the user to the next atom", () => {
    useAppStore.setState({ exchange: false });
    useAppStore.getState().setSystem("ar");
    expect(useAppStore.getState().exchange).toBe(true);
  });
});

describe("pauli toggle (configuration collapse)", () => {
  it("defaults to real physics, like every other altered-physics switch", () => {
    expect(useAppStore.getInitialState().pauli).toBe(true);
  });

  // Not a UI nicety. An exchange term exists because the wavefunction is
  // antisymmetric, and antisymmetry is what the exclusion principle is, so
  // there is no state with one and not the other for the store to hold.
  it("turning Pauli off takes exchange with it", () => {
    useAppStore.getState().setPauli(false);
    expect(useAppStore.getState().pauli).toBe(false);
    expect(useAppStore.getState().exchange).toBe(false);
  });

  it("turning Pauli back on restores real physics rather than the halfway one", () => {
    useAppStore.getState().setPauli(false);
    useAppStore.getState().setPauli(true);
    expect(useAppStore.getState().exchange).toBe(true);
  });

  it("turning exchange back on restores the cap, for the same reason", () => {
    useAppStore.setState({ pauli: false, exchange: false });
    useAppStore.getState().setExchange(true);
    expect(useAppStore.getState().pauli).toBe(true);
  });

  // The impossible combination is what the server answers 422 to. A store that
  // could hold it would eventually send it.
  it("never holds pauli off with exchange on", () => {
    const store = useAppStore.getState();
    for (const step of [
      () => store.setPauli(false),
      () => store.setExchange(false),
      () => store.setExchange(true),
      () => store.setPauli(false),
      () => store.setPauli(true),
    ]) {
      step();
      const s = useAppStore.getState();
      expect(!s.pauli && s.exchange).toBe(false);
    }
  });

  it("drops the solve, which was for a different atom entirely", () => {
    useAppStore.setState({ hf: { fake: true } as never, hfStatus: "ready" });
    useAppStore.getState().setPauli(false);
    expect(useAppStore.getState().hf).toBeNull();
    expect(useAppStore.getState().hfStatus).toBe("idle");
  });

  // A configuration carried across the switch is a different atom on one side
  // of it, and the server withholds the comparison for exactly that reason -
  // so keeping it would silently cost the user the comparison.
  it("resets the configuration to whichever ground rule is now in force", () => {
    useAppStore.setState({ config: "1s2 2s2 2p6" });
    useAppStore.getState().setPauli(false);
    expect(useAppStore.getState().config).toBeNull();
  });

  it("the collapse does not follow the user to the next atom", () => {
    useAppStore.setState({ pauli: false, exchange: false });
    useAppStore.getState().setSystem("ar");
    expect(useAppStore.getState().pauli).toBe(true);
    expect(useAppStore.getState().exchange).toBe(true);
  });
});
