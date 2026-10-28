import { useEffect, useState } from "react";
import { getAlphaTable } from "../api/client";
import type { AlphaTable } from "../api/types";
import { lutColor } from "../lib/colormap";
import { INFERNO } from "../lib/luts";
import { Badge } from "./Badge";
import { Slider } from "./Field";

const SYMBOLS = (
  "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se " +
  "Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb " +
  "Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk " +
  "Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split(" ");

/**
 * Every element under a scaled fine-structure constant: how much of its innermost
 * binding is relativity, and where the table runs out of elements altogether.
 */
export function AlphaTablePanel() {
  const [mult, setMult] = useState(1);
  const [data, setData] = useState<AlphaTable | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const t = setTimeout(() => {
      getAlphaTable(mult)
        .then((d) => live && (setData(d), setError(null)))
        .catch((e) => live && setError(e instanceof Error ? e.message : String(e)));
    }, 200);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [mult]);

  const maxShare = data
    ? Math.max(...data.shares.map((q) => (q ? q.value : 0)))
    : 1;

  return (
    <section className="config-race" aria-labelledby="alpha-title" data-tour="alpha-table">
      <h3 id="alpha-title" className="readouts-head">The periodic table at a different α</h3>
      <p className="caption">
        In atomic units the non-relativistic structure of every atom does not contain α at all, so
        changing it rescales every size and energy in metres and joules by the same factor and
        leaves chemistry's pattern alone. What it does change is relativity, through Zα, which
        grows along the table. Each cell is the share of a hydrogen-like 1s binding energy that
        relativity accounts for, from the exact Dirac formula.
      </p>
      <Slider
        label="α as a multiple of its real value"
        readout={`${mult}× α`}
        min={0.5}
        max={5}
        step={0.25}
        value={mult}
        onChange={setMult}
      />
      {error && <p className="error">{error}</p>}
      {data && (
        <>
          <div className="alpha-strip" role="img" aria-label="Relativity's share of the 1s binding per element">
            {data.shares.map((q, i) => {
              const [r, g, b] = q ? lutColor(INFERNO, Math.sqrt(q.value / maxShare)) : [60, 60, 60];
              return (
                <span
                  key={i}
                  className={q ? "alpha-cell" : "alpha-cell gone"}
                  style={{ background: `rgb(${r},${g},${b})` }}
                  title={
                    q
                      ? `${SYMBOLS[i]}: relativity is ${(100 * q.value).toFixed(2)}% of the 1s binding`
                      : `${SYMBOLS[i]}: Zα ≥ 1, no point-nucleus 1s`
                  }
                >
                  {SYMBOLS[i]}
                </span>
              );
            })}
          </div>
          <p className="caption">
            {data.z_edge < 118
              ? `At ${mult}× α the point-nucleus Dirac 1s stops existing past ${SYMBOLS[data.z_edge - 1]} (Z = ${data.z_edge}), where Zα reaches 1: the greyed elements could not hold an innermost electron at all.`
              : `At ${mult}× α every known element keeps its 1s; the edge, where Zα reaches 1, lies past oganesson.`}{" "}
            Brightness is the square root of the share, scaled to the brightest cell (a display
            choice). Gold's colour and mercury's liquidity already come from relativity in our
            world; a larger α would bring that to lighter elements.{" "}
            {data.shares[0] && <Badge provenance={data.shares[0].provenance} />}
          </p>
        </>
      )}
    </section>
  );
}
