/**
 * A number with a typographic minus, U+2212, not the hyphen toFixed() writes.
 *
 * The axes already used the true minus, so a rung reading "-13.60 eV" beside a
 * tick reading "−12 eV" was two different signs on one plot; a screen reader
 * also says "hyphen" for one and "minus" for the other.
 */
export function fixed(value: number, digits: number): string {
  return value.toFixed(digits).replace(/^-/, "−");
}
