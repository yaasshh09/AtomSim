# Phase 32: Mobile

Status: designed, not implemented.
Predecessors: Phase 30 (the instrument frontend) built the three-column layout
this reshapes, and Phase 31 (the guided tour) added the sixteen `data-tour`
anchors that a collapsed sheet would otherwise hide.

## 1. What this is

Below 900px the app currently does not render. `App.tsx` gates on viewport width
and substitutes `NarrowNotice`, which asks for a wider screen. That component's
docstring makes the argument on purpose and on the record: a stacked mobile
layout would shrink the plots past the point where an axis number is readable,
and a plot you cannot read is not an honest plot, so the app says "not here"
instead.

This phase replaces the wall with a real small-screen instrument: all seven
views, all the controls, on a phone.

The honesty argument is not being waved away, it is being taken more seriously
than the wall took it. The wall does not hold at its own threshold. At exactly
900px the centre column is 900 - 300 - 300 = 300px wide, and every plot is a
fixed-viewBox SVG at `width: 100%`, so a 680-unit viewBox renders there at
0.44x and a 12px axis label draws at about 5px. The condition the notice exists
to prevent is already met one pixel above the line where it stops applying.

The root cause is not the viewport, it is that plot geometry is written in units
that are not pixels. Section 4 fixes that everywhere, which is what makes a
phone layout honest and fixes the 900px case as a side effect.

## 2. The breakpoint and the two shells

`MIN_WIDTH` stays at 900 and keeps its meaning as the width the three-column
layout needs, but it stops being a refusal and becomes a shell selector.
`App.tsx` renders `<DesktopShell>` above it and `<MobileShell>` below it. Both
mount the same seven view components with no changes: only the chrome around
them differs.

Two shells rather than one DOM reflowed by media queries. The reasoning that put
the original gate in JS still applies in part, and the sheet forces the issue
anyway: its open state, its snap points and its drag all need JS, so a CSS-only
path is hybrid regardless and pays for a second copy of the chrome in the DOM on
both platforms. The cost accepted here is that crossing the breakpoint
mid-session remounts the tree, dropping the WebGL context and any open zoom
window. Resizing a browser across 900px is a desktop action, and a desktop user
who does it is not mid-measurement on a phone.

One breakpoint, not a phone tier and a tablet tier. A tablet at 800px gets the
mobile shell and is well served by it. The weak case is a landscape phone at
844x390, where 390px of height leaves little room once a sheet opens. The snap
points shrink there (section 3), and nothing further is done about landscape
until it proves annoying in use.

`NarrowNotice.tsx` and `NarrowNotice.test.ts` are deleted. `useViewportWidth`
moves to `lib/` since it outlives the component it was written for.

## 3. The bottom sheet

`MobileSheet.tsx` wraps the existing `InfoPanel` and `Controls` unmodified. Both
are already vertical stacks of labelled fields, which is exactly what a sheet
body wants, so this phase adds a container rather than a second set of controls.
A duplicate control panel for mobile would be two panels to keep in sync and the
first place the two platforms would drift apart on what the app can do.

Three snap points:

- **collapsed**, a 56px peek bar carrying the system name and the live n, l, m.
- **half**, 45vh.
- **full**, 90vh, body scrolls.

Under 500px of viewport height, which is every landscape phone, half becomes
35vh and full becomes 60vh. A 90vh sheet on a 390px-tall screen is a sheet that
has eaten the instrument, and the point of the half snap is that the picture
stays visible.

Half is the state the design is for. An instrument is used by changing something
and watching the picture answer, and half is the only snap where the control
under a thumb and the plot it changes are both on screen. Collapsed gives the
picture the whole viewport, full is for reading down the whole control set.

Snap selection lives in `lib/sheet.ts` as a pure function of drag offset,
velocity and viewport height, so the behaviour is unit-testable without a DOM
and the component keeps only the pointer plumbing.

Sheet state goes in the store beside `view`, `colorMode` and `nucleusMode`. It
is presentational and invalidates nothing: opening a drawer does not change any
physics, and spreading `INVALIDATED` on it would throw away a solve for a
gesture.

## 4. Plot geometry: one SVG unit is one CSS pixel

Today each plot view declares a module-level width (`const W = 640` in
`RadialView`, 680 in `LevelsView`, `SpectrumView` and `ForceLawView`, 720 in
`WhatIfView`) and renders into a `viewBox` at `width: 100%`. The SVG then scales
to whatever the container is, and every font size in it scales with it. That is
why axis text is unreadable at 300px and oversized at 1600px.

`lib/plotSize.ts` adds:

- `plotWidth(measured: number): number`, pure, clamping a measurement to a 320
  floor so a transient zero during mount cannot propagate into `NaN` scales.
- `usePlotWidth(ref)`, a `ResizeObserver` on the plot wrapper returning
  `plotWidth` of the observed content-box width.

Each view replaces its constant with the measured value and passes it to its
subcomponents (`FieldPlot` in `RadialView`, `ScreenedLadder` and `HFLadder` in
`LevelsView`, `ZoomPanel` in `SpectrumView`) as a prop, so there is one
measurement per view rather than one per plot.

This is mechanical rather than a rewrite, and the reason is worth recording:
every x coordinate in all five views is already written relative to `W`, as
`W - M.right`, `(M.left + W - M.right) / 2`, `W / 4`, `(3 * W) / 4`. No absolute
position assumes 640. Nothing in the pure exported helpers takes a width at all,
so `RadialView.test.ts`, `SpectrumView.test.ts` and the rest are unaffected.

`usePlotZoom` needs no change. It already receives width and height as arguments
and does all pointer arithmetic through `getBoundingClientRect()` ratios, so it
is correct at any width the moment the width is correct.

Height is the one place desktop appearance genuinely shifts. Today `H` scales
with `W`, so a 640x240 plot on a 1200px column renders at 1200x450. With true
size units and a fixed `H` it would render 1200x240, visibly shorter and wider.
So `H` becomes a clamped function of measured width rather than a constant,
holding desktop near what it is now while letting a phone go tall and narrow.
The clamps are per view and are expected to be tuned against screenshots rather
than derived; this spec does not fix their values.

## 5. The 3D cloud on a phone

`COUNT_CHOICES` runs to 250,000 points, which a mid-range phone GPU will not
enjoy.

No silent cap. On mobile the **default** count drops to 10,000 and all four
choices stay selectable, with a hint that the large ones will be slow. A default
is a starting value and claims nothing, so it needs no entry in
`lib/liberties.ts`. A hard cap would need one, because a cap changes the picture
from what was asked for, and that is precisely the reason not to reach for one
when a default does the job.

Two further changes, both defensive rather than presentational:

- the three.js canvas takes `dpr={[1, 2]}`, so a 3x phone does not render nine
  times the pixels of a 1x one.
- a WebGL-unavailable fallback replaces the canvas with a message. The desktop
  gate meant this path was previously unreachable on the devices most likely to
  hit it.

## 6. Touch on plots

`usePlotZoom` starts a domain pan on any `pointerdown` and calls
`preventDefault()` on wheel. On a touch screen that means a finger dragged
across a plot to scroll the page pans the plot instead, and the page does not
move. The plots would be actively hostile before they were unreadable.

The fix is a guard, not a gesture engine: drag is ignored when `pointerType` is
`touch`, the plot SVGs take `touch-action: pan-y` so the browser keeps vertical
scrolling, and the existing `ZoomControls` buttons grow to 44px targets and
become the zoom interface on touch.

Pinch to zoom and two-finger pan are deliberately not in this phase. The buttons
already cover the operation; gestures are worth building when the buttons prove
annoying against a real device, and not before.

## 7. The tour inside a sheet

Twelve of the sixteen `data-tour` anchors live in `Controls` and a thirteenth
(`state-card`) in `InfoPanel`, so thirteen of them end up inside the sheet. The
remaining three (`surface-controls`, `force-preset`, `const-sliders`) sit in
`CloudView`, `ForceLawView` and `WhatIfView` and stay in the view area.
`TourSpotlight` finds its target with
`document.querySelector` and draws nothing when the target measures zero or is
absent, which means a tour step would silently lose its ring for most of its
steps.

Rule: when a step's anchor resolves inside `.mobile-sheet` and the sheet is
collapsed, the sheet opens to half. `TourSpotlight.sync` already re-runs after
every render for reasons its docstring records, so this is a condition there
rather than new machinery.

`TourPanel` moves from inside the centre column to a fixed card above the sheet,
so the step text is not covered by the thing the step is pointing at.

`tours/anchors.test.ts` already asserts that every spotlight anchor named in the
JSON exists in the components. It is the safety net for this refactor: an anchor
dropped while moving `Controls` into a sheet fails a test rather than producing
a tour with missing rings.

## 8. Testing

No Python changes. The engine, the server and the 1360 backend tests are
untouched by this phase.

New vitest coverage on the pure pieces, which is where the logic is deliberately
put so that it can be tested without a browser:

- `plotWidth` clamping, including the transient-zero case.
- `nextSnap` snap selection across offset and velocity.
- the touch guard in `usePlotZoom`, that a `touch` pointer does not begin a pan.
- shell selection in `App`, either side of the breakpoint.

The suite is 502 tests across 46 files today. Four of them are
`NarrowNotice.test.ts` and go with the component; the remaining 498 stay green.
`tours/anchors.test.ts` guards the anchor move, and `npm run build` must pass
`tsc --noEmit`.

A Playwright pass at 390x844 and 844x390 covers what unit tests cannot see:
whether an axis label is legible, whether the sheet at half leaves usable plot,
and whether a tour step rings the control it names.

## 9. Staging

Section 4 is separable from the rest and should land first. Measured plot
geometry is a desktop improvement on its own, it fixes the 900px squeeze, and it
is the only part of this phase that touches physics-rendering code. Landing it
alone means the shell work that follows cannot be blamed for a plot that changed
shape.

After that the natural order is the shell and the sheet (sections 2 and 3), then
the cloud and touch guards (5 and 6), then the tour (7). Each is independently
shippable and independently revertible.

## 10. Out of scope

- Pinch and two-finger gestures on plots. Section 6.
- A separate tablet layout or a landscape-specific shell. Section 2.
- Any duplicate mobile control panel. Section 3.
- Offline support, installability, service workers, app manifests.
- Native wrappers.
- Rewriting `InfoPanel` or `Controls` internals. They are reparented, not
  redesigned.
