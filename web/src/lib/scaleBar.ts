/**
 * A real-size scale bar for the y = 0 planes, and one everyday anchor.
 *
 * The axes are in bohr, which is honest and means nothing to most readers. The
 * bar says the same distance in picometres (femtometres for muonic hydrogen,
 * whose 1s is 186 times smaller), and the anchor is a length anyone can check:
 * the wavelength of green light, 530 nm, the colour the eye is most sensitive to.
 */

/** Green light, nm. */
export const GREEN_NM = 530;

function unitFor(pm: number): { unit: string; perPm: number } {
  if (pm < 1) return { unit: "fm", perPm: 1000 };
  if (pm >= 1000) return { unit: "nm", perPm: 1e-3 };
  return { unit: "pm", perPm: 1 };
}

/**
 * The longest 1-2-5 length that fits in `fraction` of a frame `frameBohr` wide,
 * as {bohr, label}. The label's unit is chosen for the bar's own length.
 */
export function scaleBar(
  frameBohr: number,
  bohrPm: number,
  fraction = 0.22,
): { bohr: number; label: string } {
  const targetPm = frameBohr * bohrPm * fraction;
  const exp = Math.floor(Math.log10(targetPm));
  let best = 10 ** exp;
  for (const k of [2, 5]) if (k * 10 ** exp <= targetPm) best = k * 10 ** exp;
  const { unit, perPm } = unitFor(best);
  const shown = Number((best * perPm).toPrecision(3));
  return { bohr: best / bohrPm, label: `${shown} ${unit}` };
}

/** How many frames this wide fit in one wavelength of green light. */
export function sizeAnchor(frameBohr: number, bohrPm: number): string {
  const framePm = frameBohr * bohrPm;
  const { unit, perPm } = unitFor(framePm);
  const frame = Number((framePm * perPm).toPrecision(3));
  const count = Number(((GREEN_NM * 1000) / framePm).toPrecision(2));
  return (
    `This frame is ${frame} ${unit} across: one wavelength of green light ` +
    `(${GREEN_NM} nm) would span about ${count.toLocaleString("en-US")} of them.`
  );
}
