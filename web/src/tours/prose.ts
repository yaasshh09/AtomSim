/** Units a tour quotes. */
const UNITS = ["bohr", "eV", "nm", "pm", "MHz", "GHz", "K", "%"];

/**
 * Numbers in a body that carry a unit, and are therefore claims about a value.
 *
 * Targeting a unit rather than any numeral is what keeps "the 2p", "four
 * lobes", "step 3 of 11" and "1s2 2s2 2p6" out of the results. A lint that
 * flagged those would be turned off within a week, and a lint that is off
 * catches nothing.
 *
 * The trailing boundary is what stops "10 K" matching inside "10 Kelvins":
 * a unit has to be the whole token, not its first letter. It is also what
 * keeps the list safe to extend, since a short unit added later cannot start
 * eating the front of a longer word.
 */
export function measurementsIn(text: string): string[] {
  const units = UNITS.join("|");
  const re = new RegExp(`-?\\d+(?:\\.\\d+)?\\s(?:${units})(?![A-Za-z])`, "g");
  return text.match(re) ?? [];
}
