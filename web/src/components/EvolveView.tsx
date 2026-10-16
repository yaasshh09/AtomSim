import { useEffect, useMemo, useRef, useState } from "react";
import { VIEW_LEADS } from "../lib/explain";
import {
  MAX_COMPONENTS,
  beatPeriods,
  coefficients,
  densityBound,
  evolveDensity,
  formatFs,
  supProblem,
  type SupComponent,
} from "../lib/evolve";
import { N_CHOICES, isValidState, stateLabel } from "../lib/quantum";
import { EVOLVE_PLAYBACK_LIBERTY } from "../lib/liberties";
import { rasterize } from "../lib/rasterize";
import { isHydrogenic, isScreened } from "../lib/systemKind";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { ControlGroup, Slider } from "./Field";
import { PlaneStack } from "./PlaneStack";
import { ViewIntro } from "./ViewIntro";

/** Wall-clock seconds the slowest beat takes at 1x. Slow enough to follow one slosh. */
const SECONDS_PER_BEAT = 4;
const SPEEDS = [0.25, 1, 4];

function reducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

/** The first state not already in the mix, scanning outward from 1s. */
function nextFreeState(mix: SupComponent[]): SupComponent | null {
  for (const n of N_CHOICES) {
    for (let l = 0; l < n; l++) {
      for (let m = -l; m <= l; m++) {
        if (!mix.some((c) => c.n === n && c.l === l && c.m === m)) {
          return { n, l, m: m + 0, weight: 1, phasePi: 0 };
        }
      }
    }
  }
  return null;
}

function StateRow({
  comp,
  share,
  onChange,
  onRemove,
}: {
  comp: SupComponent;
  share: number;
  onChange: (c: SupComponent) => void;
  onRemove: (() => void) | null;
}) {
  const setNlm = (n: number, l: number, m: number) => {
    const cl = Math.min(l, n - 1);
    const cm = Math.max(-cl, Math.min(cl, m)) + 0;
    if (isValidState(n, cl, cm)) onChange({ ...comp, n, l: cl, m: cm });
  };
  return (
    <div className="evolve-row">
      <div className="evolve-row-head">
        <span className="evolve-row-name">{stateLabel(comp.n, comp.l, comp.m)}</span>
        <label>
          n{" "}
          <select value={comp.n} onChange={(e) => setNlm(Number(e.target.value), comp.l, comp.m)}>
            {N_CHOICES.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <label>
          ℓ{" "}
          <select value={comp.l} onChange={(e) => setNlm(comp.n, Number(e.target.value), comp.m)}>
            {Array.from({ length: comp.n }, (_, l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </label>
        <label>
          m{" "}
          <select value={comp.m} onChange={(e) => setNlm(comp.n, comp.l, Number(e.target.value))}>
            {Array.from({ length: 2 * comp.l + 1 }, (_, i) => i - comp.l).map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>
        {onRemove && (
          <button type="button" className="ghost" onClick={onRemove}>
            Remove
          </button>
        )}
      </div>
      <div className="evolve-row-sliders">
        <Slider
          label="Population"
          readout={`${(share * 100).toFixed(0)}%`}
          min={0}
          max={1}
          step={0.05}
          value={comp.weight}
          onChange={(weight) => onChange({ ...comp, weight })}
        />
        <Slider
          label="Phase of c"
          readout={`${comp.phasePi.toFixed(2)}π`}
          min={0}
          max={2}
          step={0.05}
          value={comp.phasePi}
          onChange={(phasePi) => onChange({ ...comp, phasePi })}
        />
      </div>
    </div>
  );
}

export function EvolveView() {
  const {
    superposition, evolve, evolveStatus, evolveProgress, error, system, systems, basis,
    setSuperposition, loadEvolve,
  } = useAppStore();
  // Ask for the fact, not its negation (see systemKind.ts): before the table
  // lands both are false, and a deep link to sodium must not fire a request.
  const hydrogenic = isHydrogenic(systems, system);
  const manyElectron = isScreened(systems, system);
  const problem = supProblem(superposition);

  useEffect(() => {
    if (hydrogenic && problem === null && !evolve && evolveStatus === "idle") {
      void loadEvolve();
    }
  }, [hydrogenic, problem, evolve, evolveStatus, loadEvolve, system, basis]);

  const [playing, setPlaying] = useState(() => !reducedMotion());
  const [speed, setSpeed] = useState(1);
  /** Bumped by Restart, so a paused movie redraws its first frame. */
  const [epoch, setEpoch] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const clockRef = useRef<HTMLSpanElement>(null);
  /** Atomic units of time. Kept out of React state: it changes every frame. */
  const tRef = useRef(0);

  const energies = useMemo(
    () => evolve?.meta.components.map((c) => c.energy.value) ?? [],
    [evolve],
  );
  const coeffs = useMemo(() => coefficients(superposition), [superposition]);
  const periods = useMemo(() => beatPeriods(energies, superposition), [energies, superposition]);
  // Fixed for the whole movie, so the density visibly breathes.
  const scale = useMemo(
    () => (evolve ? densityBound(evolve.psi, coeffs) : 0),
    [evolve, coeffs],
  );

  // A new set of grids is a new movie: start it from t = 0.
  useEffect(() => {
    tRef.current = 0;
  }, [evolve]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !evolve || problem) return;
    const res = evolve.meta.resolution;
    canvas.width = res;
    canvas.height = res;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const buf = new Float32Array(res * res);
    const slowest = periods[0];
    const unitFs = evolve.meta.time_unit_fs;

    const draw = () => {
      evolveDensity(evolve.psi, coeffs, energies, tRef.current, buf);
      ctx.putImageData(new ImageData(rasterize(buf, res, "density", scale), res, res), 0, 0);
      if (clockRef.current) {
        const t = tRef.current;
        clockRef.current.textContent = slowest
          ? `t = ${formatFs(t * unitFs)}, ${(t / slowest).toFixed(2)} of the slowest beat`
          : "Stationary: this picture is the same at every t.";
      }
    };
    draw();
    if (!playing || !slowest) return;

    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      // Capped, so a tab coming back from the background does not jump ahead.
      const dt = Math.min(now - last, 100) / 1000;
      last = now;
      tRef.current += (dt * speed * slowest) / SECONDS_PER_BEAT;
      draw();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [evolve, problem, coeffs, energies, periods, scale, playing, speed, epoch]);

  const total = superposition.reduce((s, c) => s + c.weight, 0);
  const update = (i: number, c: SupComponent) =>
    setSuperposition(superposition.map((x, j) => (j === i ? c : x)));
  const free = superposition.length < MAX_COMPONENTS ? nextFreeState(superposition) : null;
  const unitFs = evolve?.meta.time_unit_fs;

  return (
    <div className="view-wrap">
      <ViewIntro
        lead={VIEW_LEADS.evolve}
        badge={evolve ? <Badge provenance={evolve.meta.provenance} /> : undefined}
      />
      {manyElectron ? (
        <p className="hint-block">
          Time evolution needs exact stationary states, and only the hydrogen-like
          systems have them. A many-electron orbital is not an eigenstate of the whole
          atom, so a phase sum over it would look exact without being exact. Pick H,
          D, He⁺, muonic hydrogen or positronium.
        </p>
      ) : (
        <>
          <ControlGroup
            title="States in the mix"
            hint="Up to four. Populations are scaled to add up to 100%."
          >
            {superposition.map((c, i) => (
              <StateRow
                key={i}
                comp={c}
                share={total > 0 ? c.weight / total : 0}
                onChange={(next) => update(i, next)}
                onRemove={
                  superposition.length > 1
                    ? () => setSuperposition(superposition.filter((_, j) => j !== i))
                    : null
                }
              />
            ))}
            {free && (
              <button
                type="button"
                className="ghost"
                onClick={() => setSuperposition([...superposition, free])}
              >
                Add a state
              </button>
            )}
            {problem && <p className="error">{problem}</p>}
          </ControlGroup>

          <div className="view-header">
            <span className="plot-title">|Ψ(x, 0, z, t)|² [bohr⁻³]</span>
            <Badge provenance={EVOLVE_PLAYBACK_LIBERTY} />
            <button
              type="button"
              className="ghost"
              onClick={() => setPlaying((p) => !p)}
              disabled={periods.length === 0}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => {
                tRef.current = 0;
                setEpoch((e) => e + 1);
              }}
            >
              Restart
            </button>
            <div className="seg" role="group" aria-label="speed">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className={speed === s ? "seg-on" : ""}
                  onClick={() => setSpeed(s)}
                >
                  {s === 0.25 ? "¼×" : `${s}×`}
                </button>
              ))}
            </div>
          </div>
          <PlaneStack
            canvasRef={canvasRef}
            halfExtent={evolve?.meta.half_extent ?? 0}
            resolution={evolve?.meta.resolution ?? 0}
            name="atomsim-time-evolution"
          >
            {evolveStatus === "sampling" && (
              <p className="hint">computing… {(evolveProgress * 100).toFixed(0)}%</p>
            )}
            {evolveStatus === "error" && error && <p className="error">{error}</p>}
          </PlaneStack>
          {evolve && (
            <>
              <p className="caption">
                <span ref={clockRef} className="evolve-clock" />
              </p>
              <p className="caption">
                {periods.length > 0 && unitFs
                  ? `Beat periods: ${periods.map((p) => formatFs(p * unitFs)).join(", ")}. ` +
                    `At 1× the slowest one takes ${SECONDS_PER_BEAT} s of your time.`
                  : "Every populated state has the same energy, so no relative phase " +
                    "ever changes and neither does this density. Fine structure would " +
                    "split same-n states and set them beating, about α² slower."}
              </p>
              <p className="caption">
                x, z ∈ [−{evolve.meta.half_extent.toFixed(1)}, +
                {evolve.meta.half_extent.toFixed(1)}] bohr; z vertical. The inferno scale
                is fixed for all t at the largest value any frame could reach, so the
                cloud visibly breathes, and its brightness is γ-compressed at exponent
                0.5 (VISUAL LIBERTY, so faint lobes stay visible). Nothing decays here:
                a real 2p state lives 1.6 ns, millions of these beats.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}
