# Phase 3: Classical Ghost Overlay (design)

**Date:** 2026-07-16
**Branch:** `phase3-classical-ghost`
**Status:** approved, ready for planning

## 1. Purpose

Answer the question every QM course opens with, *why doesn't the atom collapse?*, 
by **showing the collapse**. Under classical electrodynamics an orbiting electron is
an accelerating charge; it radiates (Larmor), loses energy, and spirals into the
nucleus in picoseconds. Quantum mechanics forbids this. This phase overlays the
classical "ghost" of that doomed atom on the honest quantum |ψ|² cloud, animated in
real (simulated) time, under an unmissable `COUNTERFACTUAL` banner.

This is a **What-If Lab quantum-rule toggle**: "classical physics ON". It is
single-electron and self-contained, it needs none of the multi-electron machinery
that the deferred Pauli-exclusion toggle requires.

## 2. Physics (the honest core)

All quantities derive from the current state's `(n, system → Z, orbiting-particle
mass m, characteristic length a₀_sys)`. The engine is in Hartree atomic units; SI
conversions happen at the server boundary. `a₀_sys` respects the system's reduced
mass exactly as the rest of the engine already does (so muonic H, positronium, He⁺
scale correctly).

### 2.1 Bohr orbit radii: `APPROXIMATION`
```
r_n = n² · a₀_sys / Z
```
The semi-classical circular orbit for level n. Right energy scale, wrong picture, 
labeled `APPROXIMATION` (the Bohr model), not passed off as real QM.

### 2.2 Larmor radiative collapse: `COUNTERFACTUAL`
A classical charge at `r₀ = r_n` radiating per the Larmor formula obeys
`ṙ = -(4/3)·k²e⁴Z / (c³ m² r²)` (k = 1/4πε₀), which integrates to a **closed form**:
```
t_collapse = r₀³ · m² · c³ / (4 · Z · k² · e⁴)   =   r₀³ / (4 · Z · r_e² · c)
```
where `r_e = k e² / (m c²)` is the classical electron radius. In atomic units
(k=e=m=1, c = 1/α ≈ 137.036, r₀ in Bohr):
```
t_collapse = r₀³ · c³ / (4 · Z)   atomic time units      (1 a.u. time = 2.4189e-17 s)
```
**Validation anchor:** hydrogen ground state (n=1, Z=1, r₀=a₀) →
**t_collapse = 1.556×10⁻¹¹ s** (matches the textbook ~1.6×10⁻¹¹ s).
Collapses faster for higher Z (t ∝ n⁶/Z⁴), a real teaching beat.

The number is computed **exactly** under classical E&M, the rules are deliberately
wrong (`COUNTERFACTUAL`), the arithmetic is not. No approximation error term.

### 2.3 Collapse trajectory & derived readouts
Radius decays as `r(t) = r₀ · (1 − t/t_collapse)^(1/3)`. Angular velocity
`ω ∝ r^(−3/2)`; total swept angle is finite, giving a finite orbit count:
```
N_orbits = ω₀ · t_collapse / π          (ω₀ = initial angular velocity, v₀/r₀)
```
**Validation anchor:** hydrogen ground state → **N_orbits ≈ 2.05×10⁵** (~205,000
revolutions before it dies).

Readouts surfaced: `t_collapse` (in ps), initial orbital period, `N_orbits`.

### 2.4 Provenance summary
| Quantity | Tier | Method string (appended at server) |
|---|---|---|
| `r_n` Bohr radii | `APPROXIMATION` | "Bohr model r_n = n² a₀ / Z" |
| `t_collapse` | `COUNTERFACTUAL` | "Larmor radiative collapse, classical E&M, exact" |
| `N_orbits`, period | `COUNTERFACTUAL` | derived from the same classical model |
| animation slow-motion factor | `VISUAL_LIBERTY` | disclosed in the HUD (see §3) |

## 3. Honesty of the animation

Real collapse is 16 ps, unwatchable. Playback is slowed by a large factor
(auto-chosen so one collapse lasts ~5 s of wall-clock). The **live clock displays
the real simulated time in picoseconds**, never wall-clock. The slow-motion factor
is shown in the HUD as a disclosed `VISUAL_LIBERTY` (e.g. "shown at ~3×10¹¹ ×
slow-motion"). The ghost respawns at `r₀` and re-collapses on a loop so the lesson
stays on screen.

## 4. Architecture (thin, independently testable slices)

### 4.1 Engine: `src/atomsim/classical.py` (new)
Pure functions returning `Quantity` with provenance:
- `bohr_radius(n, z, system) -> Quantity` (`APPROXIMATION`)
- `collapse_time(n, z, system) -> Quantity` (`COUNTERFACTUAL`)
- `classical_ghost(n, system) -> ClassicalGhost`, aggregator dataclass bundling the
  radii, collapse time, orbit count, initial period, and the trajectory law
  parameters the client needs (`r0`, `t_collapse`, exponents are fixed constants).

Reuses `systems.py` for Z, mass, and `a₀_sys`. **Validation test** pins the two
anchors above (1.556e-11 s; N≈2.05e5) plus `r_n = n²a₀/Z`. This is engine-internal
atomic-units math; SI conversion (ps, pm) is applied at the server.

### 4.2 Server: `GET /api/classical?system=…&n=…`
Returns the ghost report through the existing `Quantity`→Pydantic mapping in
`schemas.py` (provenance survives to the browser). New `ClassicalGhostModel`.
Follows the exact pattern of the Phase-2 `/api/constants` endpoint.

### 4.3 Client
- `api/types.ts` + `api/client.ts`: `ClassicalGhost` type and `getClassical(system, n)`.
- `state/store.ts`: `classicalGhost` data slice + `ghost: boolean` toggle + loader.
  The toggle **invalidates nothing physical**, it is an overlay keyed on the
  already-known `(n, system)`, exactly like `nucleusMode`. Add to `INVALIDATED`? No.
- `lib/urlState.ts`: deep-link the toggle (`?ghost=1`), round-trip tested.
- `lib/classical.ts`: pure trajectory helper `r(τ)`, `θ(τ)`, slow-mo factor pick,
  time formatting, unit tested (no Three.js).

### 4.4 CloudView (Three.js overlay)
Behind the `ghost` toggle: Bohr ring geometry (n′=1…current, current highlighted),
the spiral trajectory line, an animated ghost point advancing on the existing render
loop, and an HTML HUD (COUNTERFACTUAL banner, live ps clock, readouts, slow-mo
disclosure). Distinct "classical" color, separate from the |ψ|² LUT. The overlay
reads `classicalGhost` from the store and self-animates; when the toggle is off it
renders nothing and does no per-frame work.

## 5. UX defaults (baked in unless changed at review)
- Ghost **loops** (respawn + re-collapse) rather than freezing "collapsed".
- Rings shown for **n′ = 1 … current**, current ring highlighted.
- Ghost starts at the **current n's** Bohr radius.
- Toggle lives in the CloudView controls next to `nucleusMode`; the overlay only
  appears in CloudView (the 3D cloud), matching the approved placement.

## 6. Testing
- **Engine:** analytic validation, `t_collapse` anchor (1.556e-11 s), `N_orbits`
  anchor (≈2.05e5), `r_n = n²a₀/Z` across a few systems (H, He⁺, muonic H).
- **Server:** endpoint shape + provenance-tier assertions (COUNTERFACTUAL /
  APPROXIMATION present and correct).
- **Client:** `classical.ts` math unit test (trajectory law, slow-mo pick,
  formatting); store + URL round-trip test.
- **Animation:** verified by live QA (server round-trip + visual), as with the
  What-If panel, no headless test of the render loop.

## 7. Out of scope (YAGNI / deferred)
- Pauli exclusion OFF, spinless electrons (separate/deferred quantum-rule toggles).
- Overlay on the 2-D plane or a dedicated view (approved placement is CloudView only).
- User-tunable slow-motion (auto-chosen; a slider is a later polish if wanted).
- Multi-electron classical dynamics.
