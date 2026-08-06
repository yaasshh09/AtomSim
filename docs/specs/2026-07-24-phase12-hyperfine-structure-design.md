# Phase 12: Hyperfine Structure (the 21 cm line)

Status: design approved 2026-07-24. Follows Phase 11 (Stark). Levels-view only,
same integration seam as Zeeman (`b_field`) and Stark (`e_field`).

## What this adds

The magnetic-dipole hyperfine interaction: the coupling of the nuclear spin **I**
to the electron's total angular momentum **J**. Each fine-structure level splits
into hyperfine levels labelled by the total angular momentum **F = I + J**,
running |I - J| to I + J. The flagship result is the hydrogen ground state: J=1/2,
I=1/2, so F = 0 or 1, and the F=1 -> F=0 transition is the 21 cm line,
1420.405751 MHz, the most famous line in radio astronomy.

## Physics (the honest boundary)

Scope for v1 is **s-states (l = 0) only**, through the **Fermi contact
interaction**. The contact term is the dominant hyperfine mechanism and the only
one that touches an s electron (its wavefunction is nonzero at the nucleus). The
orbital + spin-dipolar term that drives l > 0 hyperfine needs the <1/r^3> matrix
element and a different angular factor; it is **deferred** to a later phase and
declared absent in provenance rather than faked.

The hyperfine coupling constant of an ns level (energy, Hartree):

    A(ns) = (2/3) g_e g_I (m_e/m_p) alpha^2 (mu/m_e)^3 (Z^3 / n^3)

where

- `g_e` is the electron g-factor (2.00231930436, the measured moment, not the
  Dirac 2). Using the real moment, not a rounded 2, is legitimate physics.
- `g_I = (mu_nuc / mu_N) / I` is the nuclear g-factor of the specific nucleus.
- `m_e/m_p` is the **fixed** electron-to-proton mass ratio, because the nuclear
  magneton mu_N = e hbar / (2 m_p) is defined with the proton mass for **every**
  nucleus. (Using the nucleus's own mass here is a factor-of-2 bug for deuterium;
  it was caught and locked out by the isotope tests below.)
- `mu/m_e` is the reduced-mass ratio of the system (`System.mu_ratio`).
- `Z`, `n` are the nuclear charge and principal quantum number.

The hyperfine energy of level F, relative to the gross/fine level:

    E_hf(F) = (A/2) [ F(F+1) - I(I+1) - J(J+1) ]

For an s-state J = 1/2 (l = 0, s = 1/2). This engine reports the ns level and its
F sublevels; it does not fold in fine structure (l=0 has a single j=1/2, so there
is nothing to fold at this l anyway).

### Nuclei

Contact hyperfine needs a nucleus with a defined magnetic moment and spin. The
electron-orbiter presets:

| system | nucleus | I | mu/mu_N | note |
|--------|---------|---|---------|------|
| h      | proton  | 1/2 | +2.792847 | the 21 cm line |
| d      | deuteron| 1   | +0.857438 | F = 1/2, 3/2; split 327.384 MHz |
| t      | triton  | 1/2 | +2.978962 | 1516.70 MHz |
| he+    | alpha (He-4) | 0 | 0 | **I = 0: no hyperfine.** Honest single level. |

I = 0 is not an error and not a zero shift dressed as physics: it is the correct
statement that a spin-0 nucleus has no magnetic moment to couple to. The engine
returns the single unsplit level and says so.

Out of scope for v1, flagged (never silently wrong):

- **Positronium**: the "nucleus" is a positron with a Bohr-magneton moment, and
  annihilation contributes; the contact formula's mu_N scaling does not apply.
- **Muonic hydrogen**: the orbiter is a muon (its own g and mass enter twice);
  a real regime, but not this formula as written.
- **Generic hydrogen-like Z**: no identified nucleus, so no moment. Honest
  absence, like `System.nuclear_radius = None`.

For these the engine reports hyperfine as unavailable with the reason, rather
than returning a number.

## Fidelity

`APPROXIMATION`. Non-relativistic Fermi contact. Neglected, quantified in
provenance:

- bound-state QED beyond the free-electron g (the ~ppm radiative/recoil terms),
- the relativistic (Breit) correction, ~ (Z alpha)^2, growing with Z,
- nuclear structure (finite size / Zemach radius, the hyperfine anomaly),
- the entire l > 0 dipolar channel (deferred).

Measured residual of this formula vs experiment: ~6e-5 for hydrogen 1s, ~1e-4 for
He-3 (Z=2). The error estimate scales this up with (Z alpha)^2 so it never claims
more precision than it has.

## Validation anchors (locked in tests)

| level | engine | experiment | rel. err |
|-------|--------|------------|----------|
| H 1s (A = split) | 1420.49 MHz | 1420.4058 | 6e-5 |
| H 2s (1/n^3)     | 177.56 MHz  | 177.556   | 3e-5 |
| D 1s (split=1.5A)| 327.35 MHz  | 327.384   | 1e-4 |
| T 1s             | 1516.80 MHz | 1516.701  | 7e-5 |
| He+ (He-3) 1s    | -8666.6 MHz | -8665.65  | 1e-4 |

Five independent anchors: both I values, Z = 1 and 2, positive and negative
moments. Any regression in the mass factor, g-factor convention, or Z/n scaling
moves one of these outside tolerance and fails CI.

## Integration

Same shape as Zeeman/Stark: `/api/levels` gains a `hyperfine: bool` flag (there
is no continuous knob; the splitting is intrinsic). When on, each ns shell in the
response carries its F sublevels; the LevelsView draws an F-fan beside the shell
and a provenance Badge. No new job protocol, no invalidation of clouds/planes
(this is a Levels-only annotation, like the B and F fans).
