import { useEffect, useMemo, useRef, useState } from "react";
import { VIEW_LEADS } from "../lib/explain";
import {
  MAX_COMPONENTS,
  arrowLattice,
  beatPeriods,
  coefficients,
  currentBound,
  densityBound,
  evolveCurrent,
  evolveDensity,
  evolveField,
  formatFs,
  planeGradients,
  supProblem,
  type SupComponent,
} from "../lib/evolve";
import { N_CHOICES, isValidState, stateLabel } from "../lib/quantum";
import { EVOLVE_CURRENT_PROVENANCE, EVOLVE_PLAYBACK_LIBERTY } from "../lib/liberties";
import { rasterize, rasterizePhase } from "../lib/rasterize";
import { isHydrogenic, isScreened } from "../lib/systemKind";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { ControlGroup, Slider } from "./Field";
import { PlaneStack } from "./PlaneStack";
import { ViewIntro } from "./ViewIntro";

/** Wall-clock seconds the slowest beat takes at 1x. Slow enough to follow one slosh. */
const SECONDS_PER_BEAT = 4;
const SPEEDS = [0.25, 1, 4];
/** About this many arrows across the plane. */
const ARROWS_ACROSS = 25;
/** The canvas is drawn this many times the grid resolution, so arrows stay sharp. */
const UPSCALE = 3;
/** Arrows below this fraction of the fixed scale are left out rather than drawn as dots. */
const MIN_ARROW_FRACTION = 1e-3;

/**
 * Current arrows on the canvas. Length goes as (|j|/scale)^0.5 against a scale
 * fixed for the whole movie, the same compression the brightness gets, so the
 * flow far from the nucleus is visible beside the flow near it and a quiet
 * moment still looks quiet. Row i is z ascending, so +z goes up the canvas.
 */
function drawArrows(
  ctx: CanvasRenderingContext2D,
  at: number[],
  jx: Float64Array,
  jz: Float64Array,
  bound: number,
  res: number,
  every: number,
  k: number,
): void {
  if (bound <= 0) return;
  const longest = 0.9 * every * k;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  at.forEach((idx, p) => {
    const mag = Math.hypot(jx[p], jz[p]);
    if (mag < MIN_ARROW_FRACTION * bound) return;
    const len = longest * Math.sqrt(mag / bound);
    const ux = jx[p] / mag;
    const uy = -jz[p] / mag;
    // Pixel centres: grid index j sits at j + 0.5 on the raster.
    const cx = ((idx % res) + 0.5) * k;
    const cy = (res - 1 - Math.floor(idx / res) + 0.5) * k;
    const x0 = cx - (ux * len) / 2;
    const y0 = cy - (uy * len) / 2;
    const x1 = cx + (ux * len) / 2;
    const y1 = cy + (uy * len) / 2;
    const head = Math.min(0.4 * len, every * k * 0.3);
    const path = () => {
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.moveTo(x1 - head * (ux - 0.6 * uy), y1 - head * (uy + 0.6 * ux));
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 - head * (ux + 0.6 * uy), y1 - head * (uy - 0.6 * ux));
    };
    path();
    ctx.strokeStyle = "rgba(0, 0, 0, 0.65)";
    ctx.lineWidth = 3.5;
    ctx.stroke();
    path();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
  ctx.restore();
}

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
  const [colour, setColour] = useState<"density" | "phase">("density");
  const [showCurrent, setShowCurrent] = useState(true);
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

  const every = evolve ? Math.max(4, Math.round(evolve.meta.resolution / ARROWS_ACROSS)) : 1;
  const lattice = useMemo(
    () => (evolve ? arrowLattice(evolve.meta.resolution, every) : []),
    [evolve, every],
  );
  const grads = useMemo(() => {
    if (!evolve) return null;
    const res = evolve.meta.resolution;
    return planeGradients(evolve.psi, res, (2 * evolve.meta.half_extent) / (res - 1));
  }, [evolve]);
  const jScale = useMemo(
    () => (evolve && grads ? currentBound(evolve.psi, grads.dx, grads.dz, coeffs, lattice) : 0),
    [evolve, grads, coeffs, lattice],
  );
  // Phases are read against the first populated state's own clock: the global
  // e^{-iEt} is unobservable, and left in it would turn every hue at once.
  const refIndex = Math.max(0, superposition.findIndex((c) => c.weight > 0));
  const shifted = useMemo(
    () => energies.map((e) => e - (energies[refIndex] ?? 0)),
    [energies, refIndex],
  );

  // A new set of grids is a new movie: start it from t = 0.
  useEffect(() => {
    tRef.current = 0;
  }, [evolve]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !evolve || problem) return;
    const res = evolve.meta.resolution;
    canvas.width = res * UPSCALE;
    canvas.height = res * UPSCALE;
    const ctx = canvas.getContext("2d");
    // The physics is a res x res raster; it is drawn up with the same bilinear
    // smoothing the browser used to apply in CSS, and the arrows go on top at
    // full resolution.
    const raster = document.createElement("canvas");
    raster.width = res;
    raster.height = res;
    const rctx = raster.getContext("2d");
    if (!ctx || !rctx) return;
    const buf = new Float32Array(res * res);
    const im = new Float32Array(res * res);
    const slowest = periods[0];
    const unitFs = evolve.meta.time_unit_fs;

    const draw = () => {
      const t = tRef.current;
      let pixels;
      if (colour === "phase") {
        evolveField(evolve.psi, coeffs, shifted, t, buf, im);
        pixels = rasterizePhase(buf, im, res, scale);
      } else {
        pixels = rasterize(evolveDensity(evolve.psi, coeffs, energies, t, buf), res, "density", scale);
      }
      rctx.putImageData(new ImageData(pixels, res, res), 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(raster, 0, 0, canvas.width, canvas.height);
      if (showCurrent && grads) {
        const { jx, jz } = evolveCurrent(evolve.psi, grads.dx, grads.dz, coeffs, energies, t, lattice);
        drawArrows(ctx, lattice, jx, jz, jScale, res, every, UPSCALE);
      }
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
  }, [
    evolve, problem, coeffs, energies, periods, scale, playing, speed, epoch,
    colour, shifted, showCurrent, grads, lattice, jScale, every,
  ]);

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
            <span className="plot-title">
              {colour === "phase"
                ? "Ψ(x, 0, z, t): hue arg Ψ, brightness |Ψ|²"
                : "|Ψ(x, 0, z, t)|² [bohr⁻³]"}
            </span>
            <Badge provenance={EVOLVE_PLAYBACK_LIBERTY} />
            {showCurrent && <Badge provenance={EVOLVE_CURRENT_PROVENANCE} />}
            <div className="seg" role="group" aria-label="colour by">
              <button
                type="button"
                className={colour === "density" ? "seg-on" : ""}
                onClick={() => setColour("density")}
              >
                |Ψ|²
              </button>
              <button
                type="button"
                className={colour === "phase" ? "seg-on" : ""}
                onClick={() => setColour("phase")}
              >
                phase
              </button>
            </div>
            <button
              type="button"
              className={showCurrent ? "ghost seg-on" : "ghost"}
              aria-pressed={showCurrent}
              onClick={() => setShowCurrent((v) => !v)}
            >
              current j
            </button>
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
            bohrPm={evolve?.meta.bohr_pm}
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
              {colour === "phase" && superposition[refIndex] && (
                <p className="caption">
                  Hue is the phase of Ψ, on the same colour wheel as the point cloud, read
                  against the clock of{" "}
                  {stateLabel(
                    superposition[refIndex].n,
                    superposition[refIndex].l,
                    superposition[refIndex].m,
                  )}
                  : an overall phase e^(−iEt) cannot be measured, so it is taken out rather
                  than left to spin every colour at once. Where two states overlap, the hue
                  turning at the beat frequency is the interference that moves the density.
                </p>
              )}
              {showCurrent && (
                <p className="caption">
                  Arrows: the probability current j = Im(Ψ*∇Ψ)/μ within this plane, the flow
                  that carries the density from frame to frame. Their length goes as
                  (|j|/scale)^0.5 against a scale fixed for the whole movie, the same
                  compression as the brightness (VISUAL LIBERTY), and arrows under 0.1% of
                  the scale are left out. A single stationary state has none here: its current, if
                  m ≠ 0, circles the z axis through the plane (the Plane view's current j
                  shows it), and that out-of-plane part is not drawn.
                </p>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
