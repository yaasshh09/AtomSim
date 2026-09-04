import type { ReactNode } from "react";

/**
 * Typesetting for the plain-text physics notation that crosses the API.
 *
 * The engine writes its labels, units and provenance in ASCII: `R_6,0`,
 * `bohr^-3/2`, `mu/m_e`, `|psi|^2`, `E_HF >= E_exact`. Those strings are
 * contracts, asserted verbatim by the Python tests, so they are not the place
 * to put typography. This is: one parser that reads the notation on the way to
 * the screen and hands back parts a browser can set properly, with real
 * `<sub>`/`<sup>` in HTML and raised `<tspan>`s inside SVG.
 *
 * KaTeX is already a dependency, and is the wrong tool here. It is code-split
 * behind "Show the physics" precisely so its 260 kB never lands on a visitor
 * who only wanted an axis label, and an axis label is what most of this is.
 *
 * The one rule that keeps this honest: it never changes what a string says.
 * Subscripts, superscripts and symbol names are the same characters set at a
 * different height, so a label that was wrong stays wrong and visibly so.
 */

export type MathPart = { text: string; kind: "base" | "sub" | "sup" };

/** Symbol names the engine spells out, at word boundaries only. */
const SYMBOLS: Record<string, string> = {
  alpha: "α", beta: "β", gamma: "γ", Gamma: "Γ", delta: "δ", Delta: "Δ",
  epsilon: "ε", zeta: "ζ", eta: "η", theta: "θ", Theta: "Θ", iota: "ι",
  kappa: "κ", lambda: "λ", Lambda: "Λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π",
  Pi: "Π", rho: "ρ", sigma: "σ", Sigma: "Σ", tau: "τ", upsilon: "υ",
  phi: "φ", Phi: "Φ", chi: "χ", psi: "ψ", Psi: "Ψ", omega: "ω", Omega: "Ω",
  hbar: "ℏ", sum: "Σ", inf: "∞", infty: "∞",
};

const SYMBOL_RE = new RegExp(`\\b(${Object.keys(SYMBOLS).join("|")})\\b`, "g");

/**
 * Bases that may carry a subscript.
 *
 * A single letter is a symbol, so `m_e`, `Y_lm` and `E_HF` are subscripts. A
 * word is an identifier, so `hf_atom.py`, `sph_harm_y` and `eigh_tridiagonal`
 * are left exactly as typed. Between the two sit the spelled-out symbol names
 * and a short list of composites, which the engine does use as bases:
 * `mu_ratio`, `lambda_min`, `dE_fs`.
 */
const SUB_BASES = new Set([...Object.keys(SYMBOLS), "dE", "dV", "dr"]);

/** `->` and friends, plus bra-kets, before anything is split into parts. */
function operators(s: string): string {
  return s
    .replace(/->/g, "→")
    .replace(/<=/g, "≤")
    .replace(/>=/g, "≥")
    .replace(/!=/g, "≠")
    .replace(/\+-/g, "±")
    // The umlaut the engine has no room for in an ASCII source string.
    .replace(/\bSchrodinger\b/g, "Schrödinger")
    // Scientific notation as it is actually written. `toExponential` is where
    // most of these come from, so the digit before the e is not optional: it
    // keeps the rule off anything that is a word.
    .replace(
      /(\d)e([-+]?\d+)/g,
      (_all, d: string, exp: string) => `${d}×10^${exp.replace(/^\+/, "")}`,
    )
    // ⟨r⟩ and ⟨n'l'|r|nl⟩. A short bare token or anything with a bar in it is a
    // bracket; everything else keeps its angle brackets, because it is prose.
    .replace(/<([^<>]{1,40})>/g, (all, inner: string) =>
      inner.includes("|") || /^[A-Za-z0-9]{1,3}$/.test(inner) ? `⟨${inner}⟩` : all,
    );
}

/** Minus signs and symbol names, applied to the text of one part. */
function symbols(s: string, raised: boolean): string {
  const out = s.replace(SYMBOL_RE, (m) => SYMBOLS[m]);
  return raised ? out.replace(/-/g, "−") : out;
}

const SUP_RE = /^\^(?:\{([^}]*)\}|(-?\d+(?:\.\d+)?(?:\/\d+)?)|([A-Za-z]))/;
const SUB_RE = /^_(?:\{([^}]*)\}|([A-Za-z0-9]+(?:,[A-Za-z0-9-]+)*))/;

/**
 * The string split into runs at three heights.
 *
 * Anything the parser does not recognise stays in the base run verbatim, which
 * is what makes this safe to point at a whole sentence of provenance: the worst
 * case is that a caret survives to the screen looking exactly as it does today.
 */
export function mathParts(src: string): MathPart[] {
  const s = operators(src);
  const parts: MathPart[] = [];
  let base = "";
  const flush = () => {
    if (base) parts.push({ text: symbols(base, false), kind: "base" });
    base = "";
  };
  for (let i = 0; i < s.length; ) {
    const rest = s.slice(i);
    const sup = rest.match(SUP_RE);
    if (sup) {
      const text = sup[1] ?? sup[2] ?? sup[3];
      flush();
      parts.push({ text: symbols(text, true), kind: "sup" });
      i += sup[0].length;
      continue;
    }
    const sub = rest.match(SUB_RE);
    // A subscript needs a base to sit under, and needs not to be one more
    // underscore in a snake_case name: `column_density_m2` is a parameter, not
    // three levels of notation.
    if (sub && subscriptable(base) && !/^[_.]/.test(rest.slice(sub[0].length))) {
      const text = sub[1] ?? sub[2];
      flush();
      parts.push({ text: symbols(text, true), kind: "sub" });
      i += sub[0].length;
      continue;
    }
    // `V0`, `a0`, `n1`: a lone letter with a digit stuck to it is a subscript
    // that the engine had no character for. The letter has to stand alone for
    // this, which is what keeps it off `He2`, `log10` and `2n`.
    if (/^\d(?![0-9A-Za-z])/.test(rest) && /(?:^|[^A-Za-z0-9_])[A-Za-z]$/.test(base)) {
      const digit = s[i];
      flush();
      parts.push({ text: digit, kind: "sub" });
      i += 1;
      continue;
    }
    base += s[i];
    i += 1;
  }
  flush();
  return parts;
}

/**
 * Whether the run of letters ending `base` can carry a subscript.
 *
 * Greek counts as a letter here. Half the labels reach this already spelled
 * with the character rather than the name (`λ_NIST` is written that way in the
 * app, `mu_ratio` arrives from the engine as three ASCII letters), and a rule
 * that only knew about A-Z left every one of the first kind flat.
 */
function subscriptable(base: string): boolean {
  const word = base.match(/[A-Za-zΑ-Ωα-ω]+$/);
  if (!word) return /[⟩)\]|0-9]$/.test(base);
  return word[0].length === 1 || SUB_BASES.has(word[0]);
}

/**
 * Physics notation set as HTML.
 *
 * Takes a string rather than nodes on purpose: everything it is pointed at is
 * either an engine string or a template built from one, and keeping the input
 * flat is what lets the parser see `bohr^-3` whole instead of in three pieces.
 */
export function Notation({ children }: { children: string }) {
  return (
    <span className="math-inline">
      {mathParts(children).map((p, i) =>
        p.kind === "sup" ? (
          <sup key={i}>{p.text}</sup>
        ) : p.kind === "sub" ? (
          <sub key={i}>{p.text}</sub>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </span>
  );
}

/** How far a raised or dropped run sits off the baseline, in px at 10px type. */
const RISE = { sup: -3.5, sub: 2.5, base: 0 } as const;

/**
 * The same notation as `<tspan>`s, for an SVG `<text>`.
 *
 * SVG `dy` is relative to where the last glyph left the pen rather than to the
 * baseline, so every shift has to be undone by the run after it. Carrying the
 * current offset and emitting the difference is what keeps a label with two
 * superscripts in it from walking off the top of the plot.
 */
export function mathTspans(src: string): ReactNode[] {
  let at = 0;
  return mathParts(src).map((p, i) => {
    const dy = RISE[p.kind] - at;
    at = RISE[p.kind];
    const className = p.kind === "base" ? undefined : "exponent";
    return (
      <tspan key={i} className={className} dy={dy}>
        {p.text}
      </tspan>
    );
  });
}
