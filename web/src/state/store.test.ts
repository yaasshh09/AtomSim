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
});
