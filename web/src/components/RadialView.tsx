import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect } from "react";
import type { FieldData, Quantity, ShellPeak } from "../api/types";
import { VIEW_LEADS } from "../lib/explain";
import { HF_ORBITAL_CAPTION } from "../lib/hfModel";
import { formatHover, nearestIndex, withinPlot } from "../lib/hover";
import { informativeEndSigned, linePath, zeroCrossings } from "../lib/plot";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { Disclosure } from "./Disclosure";
import { HoverReadout, usePlotHover } from "./PlotHover";
import { ViewIntro } from "./ViewIntro";

const W = 640;
const H = 240;
const M = { top: 16, right: 16, bottom: 34, left: 56 };

/** The powers of ten I put inside [lo, hi]. */
function decades([lo, hi]: [number, number]): number[] {
  const out: number[] = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) {
    out.push(10 ** e);
  }
  return out;
}

/** I write "0.01", "1", "10" rather than "1e-2": these are bohr, and you know bohr. */
function decadeLabel(t: number): string {
  return t >= 0.01 ? String(Number(t.toFixed(2))) : t.toExponential(0);
}

/** Below this, I treat a peak's valley as too shallow to read as a shell boundary. */
const SHALLOW = 0.05;

/**
 * Enough decimals to carry the error bar to two significant figures.
 *
 * At a fixed three decimals I print helium's 0.00034 electrons displaced as
 * "0.000 ± 0.000", which is a zero standing in for a measurement that is not
 * zero, and a resolved one: the bar there is 0.00018, so the number is about
 * twice it. I let the bar set the precision, because the bar is the thing that
 * says how much precision I have.
 */
function decimalsFor(bar: number): number {
  if (!(bar > 0)) return 4;
  return Math.max(3, 1 - Math.floor(Math.log10(bar)));
}

/**
 * The disagreement in words, or my admission that it is under the noise.
 *
 * A number smaller than its own error bar is not a measurement of anything,
 * and four decimals of it would read as precision I do not have. Helium is
 * that case: 0.0003 electrons displaced against a bar of the same size.
 */
export function displacedChargeText(q: Quantity, nElectrons: number | null): string {
  const bar = q.provenance.error_estimate ?? 0;
  const of = nElectrons === null ? "" : ` of the atom's ${nElectrons}`;
  const d = decimalsFor(bar);
  if (q.value <= bar) {
    return (
      `My two models agree to within the resolution of this comparison ` +
      `(${q.value.toFixed(d)} electrons displaced, against a ${bar.toFixed(d)} bar).`
    );
  }
  return (
    `My two models disagree about where ${q.value.toFixed(d)} ± ` +
    `${bar.toFixed(d)} electrons${of} are.`
  );
}

/** One row of my shell table, where "no peak" is an answer rather than a gap. */
export function shellCells(s: ShellPeak): {
  label: string;
  gsz: string;
  hf: string;
  note: string | null;
} {
  const cell = (r: number | null) => (r === null ? "no separate peak" : r.toFixed(3));
  const shallow = [s.gsz_depth, s.hf_depth].filter(
    (d): d is number => d !== null && d < SHALLOW,
  );
  return {
    label: s.label,
    gsz: cell(s.gsz_radius),
    hf: cell(s.hf_radius),
    note:
      shallow.length === 0
        ? null
        : `barely separated: the dip before it is ${(Math.min(...shallow) * 100).toFixed(1)}% deep`,
  };
}

/**
 * A radial curve.
 *
 * `logX` puts r on a log axis, which is neither decoration nor a liberty: I
 * label the axis with the values it carries, so I hide and clip nothing. I
 * keep it because a many-electron atom is logarithmic in r and a linear axis
 * cannot show it. Argon's solve box runs to 48 bohr while its K shell sits
 * near 0.06 and its M shell near 1.4, so on a linear axis all three shells
 * pile into the first 3% of the width and I show you one spike and 45 bohr of
 * nothing. My solver's own mesh is exponential for exactly this reason.
 *
 * `title` and `blurb` are the reason this view is readable at all. The label
 * my engine ships ("R(r) radial wavefunction") names the quantity correctly
 * and tells you precisely nothing if you do not already know what it is, so I
 * put the plain sentence above the axis and keep the engine's own label beside
 * the badge where it belongs.
 */
function FieldPlot({
  field,
  title,
  blurb,
  marker,
  markerLabel,
  logX = false,
  showNodes = false,
  trimTail = false,
  overlay,
}: {
  field: FieldData;
  title: string;
  blurb: string;
  marker?: Quantity;
  markerLabel?: string;
  logX?: boolean;
  /** Have me mark where the drawn curve crosses zero. Only meaningful for signed curves. */
  showNodes?: boolean;
  /** Have me window the axis to where the curve still has amplitude. I disclose it below the plot. */
  trimTail?: boolean;
  overlay?: { field: FieldData; label: string; selfLabel: string };
}) {
  /* My solver's grid runs far past the orbital: I solve hydrogen's 3s out to
     180 bohr and it is flat past about 30, so four fifths of this plot used to
     be an axis with nothing on it. I trim the window to where the curve still
     has amplitude, and I disclose the trim under the plot with the r I cut at
     and the r the grid actually reaches, because a window drawn under the full
     grid's label would be the dishonest version of this.

     I keep the marker inside the window. ⟨r⟩ sits out in the tail for a
     diffuse state, and cutting the plot short of the number written on it
     would be worse than the empty axis this replaces. */
  const fullEnd = field.values.length;
  const trimmed = trimTail ? informativeEndSigned(field.values) : fullEnd;
  const markerEnd =
    trimTail && marker
      ? Math.min(fullEnd, nearestIndex(field.grid, marker.value * 1.08) + 1)
      : 0;
  const end = Math.max(trimmed, markerEnd);
  const grid = end < fullEnd ? field.grid.slice(0, end) : field.grid;
  const values = end < fullEnd ? field.values.slice(0, end) : field.values;
  const didTrim = end < fullEnd;

  const rMax = grid[grid.length - 1];
  const x = logX
    ? scaleLog(
        // I start at the first grid point, not zero: log has no zero, and my
        // mesh starts where it does precisely so I never need r = 0.
        [Math.max(grid[0], 1e-4), rMax],
        [M.left, W - M.right],
      )
    : scaleLinear([0, rMax], [M.left, W - M.right]);
  // I use decades only on a log axis. d3's own log ticks include every 2x, 3x,
  // and so on in each decade, and at this width they overprint into an
  // unreadable band.
  const xTicks = logX ? decades(x.domain() as [number, number]) : x.ticks(6);
  // I put both curves on one y domain, because two densities on separate
  // scales would show a disagreement neither model has.
  const overlayValues = overlay ? overlay.field.values.slice(0, end) : [];
  const all = overlay ? [...values, ...overlayValues] : values;
  const lo = Math.min(0, ...all);
  const hi = Math.max(...all);
  const y = scaleLinear([lo, hi], [H - M.bottom, M.top]).nice();
  const nodes = showNodes ? zeroCrossings(grid, values) : [];

  const hover = usePlotHover(W);
  // I snap the crosshair to a grid point, so the number in the box is my
  // engine's own sample and not something I interpolated. See lib/hover.ts for
  // why I will not bend that rule.
  const hoverIndex =
    hover.x !== null && withinPlot(hover.x, M.left, W - M.right)
      ? nearestIndex(grid, x.invert(hover.x))
      : -1;
  const hoverLines =
    hoverIndex >= 0
      ? [
          `r = ${formatHover(grid[hoverIndex])} ${field.grid_unit}`,
          `${overlay ? overlay.selfLabel : "value"} = ${formatHover(values[hoverIndex])}`,
          ...(overlay && hoverIndex < overlayValues.length
            ? [`${overlay.label} = ${formatHover(overlayValues[hoverIndex])}`]
            : []),
        ]
      : [];

  return (
    <figure className="plot">
      <figcaption>
        <span className="plot-lead">{title}</span>
        <span className="plot-blurb">{blurb}</span>
        <span className="plot-provenance">
          {field.label} [{field.unit}] <Badge provenance={field.provenance} />
          {overlay && (
            <span className="legend-inline">
              <span className="swatch-line" /> {overlay.selfLabel}
              <span className="swatch-line dashed" /> {overlay.label}
            </span>
          )}
        </span>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        className="plot-hoverable"
        ref={hover.ref}
        onPointerMove={hover.onPointerMove}
        onPointerLeave={hover.onPointerLeave}
      >
        <line
          x1={M.left} y1={H - M.bottom} x2={W - M.right} y2={H - M.bottom}
          className="axis"
        />
        <line x1={M.left} y1={M.top} x2={M.left} y2={H - M.bottom} className="axis" />
        {xTicks.map((t) => (
          <g key={t} transform={`translate(${x(t)},${H - M.bottom})`}>
            <line y2="5" className="axis" />
            <text y="18" textAnchor="middle" className="tick">
              {logX ? decadeLabel(t) : t}
            </text>
          </g>
        ))}
        {y.ticks(4).map((t) => (
          <g key={t} transform={`translate(${M.left},${y(t)})`}>
            <line x2="-5" className="axis" />
            <text x="-8" dy="0.32em" textAnchor="end" className="tick">
              {t.toPrecision(2)}
            </text>
          </g>
        ))}
        {lo < 0 && (
          <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} className="zero" />
        )}
        <path d={linePath(grid, values, x, y)} className="curve" />
        {overlay && (
          <path
            d={linePath(overlay.field.grid.slice(0, end), overlayValues, x, y)}
            className="curve curve-overlay"
          />
        )}
        {/* I draw the nodes on the axis rather than through the curve: the
            mark points at a place on r, and a full-height rule there would
            read as another quantity being plotted. */}
        {nodes.map((r) => (
          <g key={r} className="node-mark">
            <line x1={x(r)} x2={x(r)} y1={y(0) - 5} y2={y(0) + 5} />
            <circle cx={x(r)} cy={y(0)} r={2.5} />
          </g>
        ))}
        {marker && (
          <g className="marker-group">
            <line
              x1={x(marker.value)} x2={x(marker.value)} y1={M.top} y2={H - M.bottom}
              className="marker"
            />
            <text x={x(marker.value) + 4} y={M.top + 11} className="tick marker-text">
              {markerLabel ?? "⟨r⟩"} = {marker.value.toFixed(2)}
            </text>
          </g>
        )}
        <text
          x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle" className="tick"
        >
          r [{field.grid_unit}]{logX ? " (log)" : ""}
        </text>
        {hoverIndex >= 0 && (
          <HoverReadout
            px={x(grid[hoverIndex])}
            py={y(values[hoverIndex])}
            top={M.top}
            bottom={H - M.bottom}
            lines={hoverLines}
            width={W}
            rightMargin={M.right}
          />
        )}
      </svg>
      {(nodes.length > 0 || didTrim) && (
        <p className="plot-foot">
          {nodes.length > 0 && (
            <>
              I marked {nodes.length} radial node{nodes.length === 1 ? "" : "s"} on
              the axis, where the curve changes sign. The count is n − ℓ − 1, always.{" "}
            </>
          )}
          {didTrim && (
            <>
              {/* I say "the data I was sent", not "the solve": my server now
                  windows its own output to the orbital, so the last grid point
                  here is the end of the payload and not the wall of the solve
                  box. Claiming otherwise would have me asserting a number
                  nobody ever told me. The badge carries both radii. */}
              I stop the axis at {formatHover(rMax)} {field.grid_unit}, past which
              the curve is under a thousandth of its peak; the data I was sent
              runs to {formatHover(field.grid[field.grid.length - 1])}.
            </>
          )}
        </p>
      )}
    </figure>
  );
}

export function RadialView() {
  const {
    n, l, system, radial, stateInfo, loadRadial, model, config, exchange, pauli, compare,
  } = useAppStore();
  useEffect(() => {
    void loadRadial();
    // I depend on the Hartree-Fock inputs too: each of them names a different
    // solve, so a curve I fetched under one and left on screen under another
    // would be the stale render my store's INVALIDATED block exists to
    // prevent, arriving by a different door.
    //
    // `compare` is here for the opposite reason. It names no different solve,
    // but it does name a field the payload is missing, and my setCompare drops
    // the payload rather than the atom. Without it here I would sit on
    // "loading" forever after the toggle.
  }, [n, l, system, model, config, exchange, pauli, compare, loadRadial]);

  const intro = (
    <ViewIntro lead={VIEW_LEADS.radial}>
      <Disclosure summary="Why the two plots disagree at the nucleus">
        <p className="caption">
          R(r) is the wavefunction's radial part. For an s orbital it is at its
          largest right at r = 0, so I start the top plot high.
        </p>
        <p className="caption">
          P(r) is the probability of finding the electron in a thin shell at
          radius r, and it is R(r)² multiplied by the surface area of that
          shell, 4πr². At r = 0 the shell has no area, so however large R is
          there, P is zero. That is not a contradiction and not a numerical
          artefact of mine: the electron is most likely to be found{" "}
          <em>near</em> the nucleus, and exactly at it there is no room to be
          found in.
        </p>
        <p className="caption">
          It is P(r) you should integrate, and my second plot is the one whose
          area means something. ⟨r⟩, which I mark on it, is its average, and it
          sits to the right of the peak because the curve has a long tail.
        </p>
      </Disclosure>
      <Disclosure summary="How to read the wiggles">
        <p className="caption">
          Each place R(r) crosses zero is a radial node, a sphere the electron
          is never found on. There are always n − ℓ − 1 of them, so 3s has two,
          3p has one and 3d has none. Raise ℓ at fixed n and you trade the
          wiggles for angular structure, which is what I draw in the 3-D cloud
          view.
        </p>
      </Disclosure>
    </ViewIntro>
  );

  if (!radial) {
    return (
      <div className="view-wrap">
        {intro}
        <p className="hint-block">I am loading the radial functions…</p>
      </div>
    );
  }

  return (
    <div className="view-wrap">
      {intro}
      <FieldPlot
        field={radial.r_wavefunction}
        title="R(r), the radial wavefunction"
        blurb="The amplitude, sign and all. It can go negative; I keep that sign because it is real, and it is what the nodes are."
        showNodes
        trimTail
      />
      <FieldPlot
        field={radial.radial_probability}
        title="P(r), where the electron actually is"
        blurb="Probability per unit radius. Its area under any stretch of r is the chance of finding the electron in that stretch."
        marker={stateInfo?.mean_radius ?? undefined}
        markerLabel="⟨r⟩"
        trimTail
      />
      {model === "hf" && <p className="hint-block">{HF_ORBITAL_CAPTION}</p>}
      {radial.total_density && (
        <>
          {/* I keep the primary curve on the model your radio selected, so the
              plot you were already looking at does not move under you. */}
          <FieldPlot
            field={radial.total_density}
            title="D(r), the whole atom's electrons"
            blurb="Every occupied orbital, summed. Each bump is a shell, and this is the one curve here an experiment can measure directly."
            logX
            overlay={
              radial.density_comparison
                ? {
                    field:
                      model === "hf"
                        ? radial.density_comparison.gsz
                        : radial.density_comparison.hf,
                    label: model === "hf" ? "screened (GSZ)" : "Hartree-Fock",
                    selfLabel: model === "hf" ? "Hartree-Fock" : "screened (GSZ)",
                  }
                : undefined
            }
          />
          <p className="hint-block">
            Neon has two peaks and argon has three. That is the periodic table,
            drawn: count the bumps and you have counted the shells. My whole
            curve integrates to {radial.system.n_electrons ?? "N"}, which is
            what makes it a density rather than a shape.
          </p>
          <Disclosure summary="How much of a shell is really in one peak">
            <p className="hint-block">
              Integrate across one peak in r and you get roughly how many
              electrons it holds. Roughly, because neighbouring shells overlap
              and the dip between two of them is not a wall: cut argon at its
              minima and I give you a K shell near 2.2, not 2, under either
              model.
            </p>
          </Disclosure>
          <Disclosure summary="What the log axis costs" tone="caveat">
            <p className="hint-block">
              I put r on a log axis because the shells are decades apart in
              size, not to flatter the shape. The price is that the areas you
              can see here are not those integrals: my axis stretches the inner
              shell and squeezes the outer one, so read the peaks, and trust a
              number only where I have written it down. The plots above are one
              orbital out of the basis I summed this from.
            </p>
          </Disclosure>
          {model === "gsz" && (
            <Disclosure summary="What a fitted model gets right, and where it gives out">
              <p className="hint-block">
                GSZ was fitted to reproduce a potential rather than a density,
                and every shell I solve here sees the same one. That turns out
                to cost less than it sounds: turn the comparison on and my two
                models place under 1.5% of the electrons differently for every
                atom they both cover, because the fit was made against
                Hartree-Fock in the first place. Where the fitted model gives
                out is the energy, not the shape.
              </p>
            </Disclosure>
          )}
        </>
      )}
      {radial.density_comparison && (
        <div className="compare-block">
          <p className="hint-block">
            {displacedChargeText(
              radial.density_comparison.displaced_charge,
              radial.system.n_electrons ?? null,
            )}{" "}
            <Badge provenance={radial.density_comparison.provenance} />
          </p>
          <table className="shell-table">
            <caption>
              Shell peak radii [bohr]: where each of my models puts the middle of each shell
            </caption>
            <thead>
              <tr>
                <th>shell</th>
                <th>GSZ</th>
                <th>Hartree-Fock</th>
              </tr>
            </thead>
            <tbody>
              {radial.density_comparison.shells.map(shellCells).map((c) => (
                <tr key={c.label}>
                  <th scope="row">{c.label}</th>
                  <td>{c.gsz}</td>
                  <td>{c.hf}</td>
                  {c.note && <td className="shell-note">{c.note}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
