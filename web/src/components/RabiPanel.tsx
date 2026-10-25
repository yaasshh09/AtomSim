import { scaleLinear } from "d3-scale";
import { useEffect, useState } from "react";
import { getRabi } from "../api/client";
import type { RabiDrive } from "../api/types";
import { formatSeconds } from "../lib/cascade";
import { fixed } from "../lib/format";
import { usePlotWidth } from "../lib/plotSize";
import { formatIntensity, RABI_TRANSITIONS, type RabiTransition } from "../lib/rabi";
import { Badge } from "./Badge";
import { Choice, Slider } from "./Field";
import { Disclosure } from "./Disclosure";

const PAD = { left: 44, right: 16, top: 16, bottom: 30 };
/** Slider drags fire on every step; wait this long for the hand to stop. */
const DEBOUNCE_MS = 200;

const mhz = (radPerS: number) => radPerS / (2 * Math.PI * 1e6);

/**
 * Shine a laser on one transition and watch the population slosh: Rabi
 * oscillations, damped by the upper level's real decay. Hydrogen-like only;
 * LevelsView mounts it on that branch alone.
 */
export function RabiPanel({ system }: { system: string }) {
  const [transition, setTransition] = useState<RabiTransition>("1s-2p");
  const [logI, setLogI] = useState(6);
  const [detuning, setDetuning] = useState(0);
  const [data, setData] = useState<RabiDrive | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { width: W, ref } = usePlotWidth(280);

  useEffect(() => {
    let live = true;
    const t = RABI_TRANSITIONS.find((x) => x.value === transition)!;
    const timer = setTimeout(() => {
      getRabi({ ...t, intensity: 10 ** logI, detuning, system }).then(
        (r) => {
          if (!live) return;
          setData(r);
          setError(null);
        },
        (e) => live && setError(e instanceof Error ? e.message : String(e)),
      );
    }, DEBOUNCE_MS);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [transition, logI, detuning, system]);

  return (
    <section className="config-race" aria-labelledby="rabi-title" ref={ref}>
      <h3 id="rabi-title" className="readouts-head">
        Drive it with a laser: Rabi oscillations
      </h3>
      <p className="caption">
        Light tuned to a transition does not just kick the atom up. It swings the
        population back and forth between the two levels at the Rabi frequency, which
        grows with the light's field strength. Meanwhile the upper level keeps decaying,
        which washes the swinging out into a steady state.
      </p>
      <Choice
        legend="Transition"
        options={RABI_TRANSITIONS.map((t) => ({
          value: t.value,
          label: t.value.replace("-", " ↔ "),
          hint: t.hint,
        }))}
        value={transition}
        onChange={setTransition}
      />
      <Slider
        label="Laser intensity"
        readout={formatIntensity(10 ** logI)}
        min={0}
        max={14}
        step={0.25}
        value={logI}
        onChange={setLogI}
      />
      <Slider
        label="Detuning from resonance"
        readout={`${fixed(detuning, 0)} MHz`}
        min={-1000}
        max={1000}
        step={10}
        value={detuning}
        onChange={setDetuning}
        atRest={detuning === 0}
      />
      {error && <p className="error">{error}</p>}
      {data && <RabiBody data={data} W={W} />}
    </section>
  );
}

function RabiBody({ data, W }: { data: RabiDrive; W: number }) {
  const omega = data.rabi_frequency.value;
  const gamma = data.decay_rate.value;
  const ratio = omega / gamma;
  const lostEnd = data.lost.values[data.lost.values.length - 1];
  return (
    <>
      <p className="caption">
        Rabi frequency Ω/2π = <strong>{mhz(omega).toPrecision(3)} MHz</strong>{" "}
        <Badge provenance={data.rabi_frequency.provenance} />, against a decay rate
        Γ/2π = {mhz(gamma).toPrecision(3)} MHz.{" "}
        {ratio > 3
          ? "The drive beats the decay, so several full swings show before they fade."
          : ratio > 0.3
            ? "Drive and decay are evenly matched: one or two swings, then a steady state."
            : "The decay wins: the atom barely lifts off the ground before it falls back."}{" "}
        {data.steady_excited &&
          `It settles with ${fixed(100 * data.steady_excited.value, 1)}% of the atom in the upper level, never more than half.`}
        {!data.steady_excited &&
          ` Only ${fixed(100 * data.return_fraction.value, 0)}% of decays come back to the driven level, so the laser slowly pumps the atom out of the pair: ${fixed(100 * lostEnd, 0)}% is gone by the end of the plot.`}
      </p>
      <RabiPlot data={data} W={W} />
      <Disclosure summary="What the laser model leaves out" tone="caveat">
        <p className="caption">
          Two sublevels with the same m, coupled by light polarized along z. The
          rotating-wave approximation drops terms of relative size Ω/ω₀, here{" "}
          {data.rwa_parameter.value.toExponential(1)}; past 0.1 the server refuses rather
          than answer wrongly. The laser is a perfect single frequency, the atom stands
          still (no Doppler shift), and nothing ionizes it. Decays to levels the laser does
          not touch are counted as lost, not followed further.
        </p>
      </Disclosure>
    </>
  );
}

function RabiPlot({ data, W }: { data: RabiDrive; W: number }) {
  const H = 220;
  const t = data.excited.grid;
  const tMax = t[t.length - 1];
  const x = scaleLinear().domain([0, tMax]).range([PAD.left, W - PAD.right]);
  const y = scaleLinear().domain([0, 1]).range([H - PAD.bottom, PAD.top]);
  const path = (v: number[]) =>
    v.map((p, i) => `${i ? "L" : "M"}${x(t[i]).toFixed(1)},${y(p).toFixed(1)}`).join("");
  const ticks = x.ticks(4);
  return (
    <figure className="plot">
      <figcaption>
        <span className="plot-lead">Chance the atom is in the upper level</span>
        <span className="plot-blurb">
          Solid: with the real decay. Dashed: the same drive if the upper level never
          decayed{data.steady_excited ? "" : "; the faint curve is the population lost from the pair"}.
        </span>
        <span className="plot-provenance">
          P_e(t) <Badge provenance={data.excited.provenance} />
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Upper-level population against time">
        <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} />
        <line className="axis" x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} />
        {[0, 0.5, 1].map((v) => (
          <text key={v} className="tick" x={PAD.left - 6} y={y(v) + 4} textAnchor="end">
            {v}
          </text>
        ))}
        {ticks.map((s) => (
          <text key={s} className="tick" x={x(s)} y={H - 10} textAnchor="middle">
            {formatSeconds(s, 2)}
          </text>
        ))}
        <path d={path(data.excited_undamped.values)} className="curve curve-overlay" />
        {!data.steady_excited && (
          <path d={path(data.lost.values)} fill="none" stroke="var(--muted)" strokeWidth={1} />
        )}
        <path d={path(data.excited.values)} className="curve" strokeWidth={2} />
      </svg>
    </figure>
  );
}
