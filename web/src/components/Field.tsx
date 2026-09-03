import type { ReactNode } from "react";

/**
 * The control vocabulary shared across the instrument views.
 *
 * Before this, every view invented its own: LevelsView packed four controls
 * into the same flex row as its title, and SpectrumView stacked nine bare
 * checkboxes that revealed one another with nothing to say which layer you
 * were in. The physics under them was fine. The arrangement was why nobody
 * could find it.
 *
 * Three rules hold across everything here. A control says what it does in
 * words, not in a symbol. A number carries its meaning beside it, so "20 T"
 * never appears without something that says 20 T is a lot. A control that
 * cannot act right now gets disabled and explained rather than hidden: a
 * control that disappears is one somebody goes hunting for.
 */

/** A titled group of related controls, with an optional one-line purpose. */
export function ControlGroup({
  title,
  hint,
  children,
  tone = "plain",
}: {
  title: string;
  hint?: string;
  children: ReactNode;
  /** `active` tints the rail when this group is doing something. */
  tone?: "plain" | "active";
}) {
  return (
    <section className={`ctl-group ctl-group-${tone}`}>
      <h3 className="ctl-group-title">{title}</h3>
      {hint && <p className="ctl-group-hint">{hint}</p>}
      <div className="ctl-group-body">{children}</div>
    </section>
  );
}

/**
 * A range input with its value and its meaning on the same line.
 *
 * `readout` is the number in its own units, and `anchor` is what that number
 * is comparable to. They stay separate parameters because they are separate
 * claims: the readout comes from the app, the anchor is a fact about the
 * world, and only the readout may ever be derived from a solve.
 */
export function Slider({
  label,
  readout,
  anchor,
  min,
  max,
  step,
  value,
  onChange,
  disabled = false,
  atRest = false,
  tourId,
}: {
  label: string;
  readout: string;
  anchor?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  /** True when the slider sits where nothing has been applied. */
  atRest?: boolean;
  tourId?: string;
}) {
  return (
    <label className="ctl-slider" data-tour={tourId} data-rest={atRest || undefined}>
      <span className="ctl-slider-head">
        <span className="ctl-slider-label">{label}</span>
        <span className="ctl-slider-readout">{readout}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {anchor && <span className="ctl-slider-anchor">{anchor}</span>}
    </label>
  );
}

/**
 * A checkbox with a plain-English label and a line saying what it does.
 *
 * `why` shows whenever it is given, in both states. A hint that only appears
 * once a box is ticked cannot help anyone decide whether to tick it, which is
 * the moment the hint is for.
 */
export function Toggle({
  label,
  checked,
  onChange,
  why,
  disabled = false,
  disabledReason,
  tourId,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  why?: string;
  disabled?: boolean;
  /** Shown in place of `why` when the control cannot act. */
  disabledReason?: string;
  tourId?: string;
}) {
  const note = disabled && disabledReason ? disabledReason : why;
  return (
    <div className={`ctl-toggle${disabled ? " ctl-toggle-off" : ""}`} data-tour={tourId}>
      <label className="ctl-toggle-label">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span>{label}</span>
      </label>
      {note && <p className="ctl-toggle-why">{note}</p>}
    </div>
  );
}

/** One option in a {@link Choice}. */
export interface ChoiceOption<T extends string> {
  value: T;
  label: string;
  /** Shown under the row once this option is the live one. */
  hint?: string;
  disabled?: boolean;
  /** Why this option cannot be picked. Shown when the option is disabled. */
  disabledReason?: string;
}

/**
 * A segmented picker for a set of options that used to be mutually exclusive
 * by accident.
 *
 * LevelsView is the case that motivated it. Its magnifier panel could show
 * fine structure, a Zeeman fan, a Stark manifold or a hyperfine split, and
 * which one you got fell out of a chain of nested ternaries: turn on hyperfine
 * with the electric field already up, and the hyperfine toggle silently did
 * nothing. Exclusivity that is real should be visible, so it is a picker now,
 * and an option that cannot run says so on its face.
 */
export function Choice<T extends string>({
  legend,
  options,
  value,
  onChange,
  tourId,
}: {
  legend: string;
  options: ChoiceOption<T>[];
  value: T;
  onChange: (v: T) => void;
  tourId?: string;
}) {
  const live = options.find((o) => o.value === value);
  return (
    <div className="ctl-choice" data-tour={tourId}>
      <span className="ctl-choice-legend">{legend}</span>
      <div className="ctl-choice-row" role="group" aria-label={legend}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            className={`ctl-choice-btn${value === o.value ? " ctl-choice-on" : ""}`}
            aria-pressed={value === o.value}
            disabled={o.disabled}
            title={o.disabled ? o.disabledReason : undefined}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
      {live?.hint && <p className="ctl-choice-hint">{live.hint}</p>}
      {live?.disabled && live.disabledReason && (
        <p className="ctl-choice-hint ctl-choice-blocked">{live.disabledReason}</p>
      )}
    </div>
  );
}
