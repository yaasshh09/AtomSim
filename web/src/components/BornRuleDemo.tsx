import { scaleLinear } from "d3-scale";
import { useEffect, useMemo, useRef, useState } from "react";
import { createSampleJob, getChannel, watchJob } from "../api/client";
import type { FieldData } from "../api/types";
import { exactCdf, histogram, ksDistance, radii, typicalKs } from "../lib/bornRule";
import { manyElectronParams } from "../lib/hfModel";
import { linePath } from "../lib/plot";
import { informativeEnd } from "../lib/spark";
import { useAppStore } from "../state/store";

/** Draws fetched per run; the reveal never needs more than this. */
const POOL = 20_000;
const BINS = 40;
const SPEEDS = [
  { label: "1 / s", perSecond: 1 },
  { label: "20 / s", perSecond: 20 },
  { label: "1000 / s", perSecond: 1000 },
];

/**
 * Measure the electron's position once, then again, then again.
 *
 * Every point revealed is an independent draw from |psi|^2, taken from a sample
 * job, so the histogram is what an experimenter with a perfect detector and a
 * freshly prepared atom for every shot would record. It starts as noise and
 * settles onto P(r), and the KS distance says how fast: like 1/sqrt(N), which
 * is the signature of honest draws.
 */
export function BornRuleDemo({ pr, width }: { pr: FieldData; width: number }) {
  const store = useAppStore();
  const { n, l, m, basis, system } = store;
  const [pool, setPool] = useState<Float32Array | null>(null);
  const [shown, setShown] = useState(0);
  const [speed, setSpeed] = useState<number | null>(null);
  const [run, setRun] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const acc = useRef(0);
  const params = manyElectronParams(store);
  const paramKey = JSON.stringify([n, l, m, basis, system, params]);

  // A different state is a different experiment: start it from zero.
  useEffect(() => {
    setPool(null);
    setShown(0);
    setSpeed(null);
  }, [paramKey]);

  // The pool is finite; at its end the run stops rather than wrapping around.
  useEffect(() => {
    if (shown >= POOL) setSpeed(null);
  }, [shown]);

  const ensurePool = async (): Promise<Float32Array | null> => {
    if (pool) return pool;
    setLoading(true);
    setError(null);
    try {
      const job = await createSampleJob({
        n, l, m, basis, system, count: POOL, seed: 1000 + run, ...params,
      });
      await watchJob(job.id, () => {});
      const pos = await getChannel(job.id, "positions");
      setPool(pos);
      return pos;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (speed === null || !pool) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      acc.current += ((now - last) / 1000) * speed;
      last = now;
      const add = Math.floor(acc.current);
      if (add > 0) {
        acc.current -= add;
        setShown((s) => Math.min(s + add, POOL));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [speed, pool]);

  const end = Math.max(2, informativeEnd(pr.values));
  const grid = pr.grid.slice(0, end);
  const values = pr.values.slice(0, end);
  const rMax = grid[grid.length - 1];
  const cdf = useMemo(() => exactCdf(pr.grid, pr.values), [pr]);
  const rs = useMemo(() => (pool ? radii(pool, shown) : new Float64Array()), [pool, shown]);
  const hist = useMemo(() => histogram(rs, rMax, BINS), [rs, rMax]);
  const ks = rs.length > 0 ? ksDistance(rs, pr.grid, cdf) : null;

  const H = 200;
  const pad = { l: 44, r: 12, t: 12, b: 30 };
  const x = scaleLinear([0, rMax], [pad.l, width - pad.r]);
  const yMax = Math.max(...values, ...hist) * 1.08 || 1;
  const y = scaleLinear([0, yMax], [H - pad.b, pad.t]);
  const bw = (x(rMax) - x(0)) / BINS;

  const measureOne = async () => {
    if (await ensurePool()) setShown((s) => Math.min(s + 1, POOL));
  };
  const play = async (perSecond: number) => {
    if (await ensurePool()) setSpeed((cur) => (cur === perSecond ? null : perSecond));
  };
  const reset = () => {
    setPool(null);
    setShown(0);
    setSpeed(null);
    setRun((r) => r + 1);
  };

  return (
    <section className="born-rule" aria-labelledby="born-title">
      <h3 id="born-title" className="readouts-head">
        The Born rule, one measurement at a time
      </h3>
      <p className="caption">
        Each click finds the electron once: one position, drawn from |ψ|². One shot says
        almost nothing. Keep measuring and the histogram of r settles onto P(r), the curve
        above, and nothing about any single shot told you it would.
      </p>
      <div className="view-header">
        <button type="button" className="ghost" onClick={() => void measureOne()} disabled={loading}>
          Measure once
        </button>
        <div className="seg" role="group" aria-label="measurement rate">
          {SPEEDS.map((s) => (
            <button
              key={s.perSecond}
              type="button"
              className={speed === s.perSecond ? "seg-on" : ""}
              onClick={() => void play(s.perSecond)}
              disabled={loading}
            >
              {s.label}
            </button>
          ))}
        </div>
        <button type="button" className="ghost" onClick={reset} disabled={shown === 0}>
          Start over
        </button>
        {loading && <span className="hint">preparing {POOL.toLocaleString()} atoms…</span>}
      </div>
      {error && <p className="error">{error}</p>}
      <svg width={width} height={H} role="img" aria-label="histogram of measured radii over P(r)">
        {hist.map((h, i) =>
          h > 0 ? (
            <rect
              key={i}
              className="born-bar"
              x={x(0) + i * bw}
              y={y(h)}
              width={Math.max(bw - 1, 1)}
              height={y(0) - y(h)}
            />
          ) : null,
        )}
        <path className="born-curve" d={linePath(grid, values, x, y)} />
        <line className="born-axis" x1={x(0)} x2={x(rMax)} y1={y(0)} y2={y(0)} />
        {x.ticks(5).map((t) => (
          <text key={t} x={x(t)} y={H - 10} textAnchor="middle" className="born-tick">
            {t}
          </text>
        ))}
        <text x={width - pad.r} y={H - 10} textAnchor="end" className="born-tick">
          r [bohr]
        </text>
      </svg>
      <p className="caption">
        {shown === 0
          ? "No measurements yet."
          : `${shown.toLocaleString()} measurement${shown > 1 ? "s" : ""}, the last at r = ${rs[
              rs.length - 1
            ].toFixed(2)} bohr. KS distance to P(r): ${ks?.toFixed(4)}; honest draws typically sit near ${typicalKs(
              shown,
            ).toFixed(4)} at this N and keep shrinking like 1/√N.`}
      </p>
      <p className="caption">
        Every point is a real draw from |ψ|², and the engine's own tests hold those draws to
        the exact distribution. One idealization is assumed rather than modelled: a real
        position measurement disturbs the atom, so each shot here stands for a freshly
        prepared one. Bars are counts per bohr divided by N, so their area is 1, the same as
        the curve's.
      </p>
    </section>
  );
}
