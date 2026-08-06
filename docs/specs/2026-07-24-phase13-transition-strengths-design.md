# Phase 13: Transition Strengths (line intensities, lifetimes)

Status: design approved 2026-07-24. Follows Phase 12 (hyperfine). **Engine-only
this phase**, the server/client surfacing of intensities in the Spectrum view
is a deliberate later increment. This lands the physics core, validated.

## What this adds

The spectrum today has energies and wavelengths but no **intensities**. This
adds the electric-dipole transition strengths for hydrogen-like atoms:

- the **radial dipole matrix element** R = ∫ R_{n'l'}(r) · r · R_{nl}(r) · r² dr,
- the **absorption oscillator strength** f (dimensionless),
- the **Einstein A coefficient** (spontaneous emission rate, s⁻¹),
- the **radiative lifetime** τ of an upper level (1 / Σ A).

## Physics

The radial functions R_nl are the EXACT closed forms already in `hydrogen.py`.
The dipole integral has a closed form (Gordon), but we integrate the exact
functions by high-order Gauss quadrature and estimate the error by doubling the
node count, **NUMERICAL** tier, validated against exact analytic anchors. This
matches the project's radial-solver pattern (numerics checked against analytics).

Formulas (atomic units; l is the lower level's l, l_max = max(l, l')):

    f_abs(nl -> n'l') = (2/3) (E_{n'} - E_{nl}) (l_max / (2l+1)) |R|²
    A_emit(n'l' -> nl) = (4/3) α³ (E_{n'} - E_{nl})³ (l_max / (2l'+1)) |R|²   [1/t_au]
    τ(n'l') = 1 / Σ_{nl} A_emit(n'l' -> nl)

A is converted to s⁻¹ by dividing by the atomic time unit t_au = ℏ/E_h. The
(2l+1) in f and the (2l'+1) in A are the initial and final spatial degeneracies;
the algebra makes A independent of the lower degeneracy, as it must be.

Selection rule: electric dipole requires l' = l ± 1. Δl = 0 or |Δl| ≥ 2 returns a
strength of exactly zero, tagged as a disclosed selection-rule zero (not a
silent zero). Same-level or wrong-direction energy pairs raise.

## Fidelity

`NUMERICAL`. The wavefunctions are EXACT; the quadrature carries a
node-doubling error estimate (~1e-10 relative for n ≤ 6 with a few hundred
Gauss-Legendre nodes on a range scaled to the outer turning region). f and A are
therefore accurate to the quadrature error. What is NOT modelled, and stated in
provenance: fine-structure/relativistic corrections to the rates, QED, and
multi-electron effects (this is one-electron hydrogenic).

## Validation anchors (locked in tests)

| quantity | engine | reference | source |
|----------|--------|-----------|--------|
| f(1s→2p) | 0.4162 | 0.41620 | Bethe-Salpeter |
| f(1s→np) | decreasing in n |, | monotonicity |
| A(2p→1s) | 6.27e8 s⁻¹ | 6.27e8 | NIST ASD |
| τ(2p)    | 1.60 ns | 1.596 ns | NIST ASD |
| Δl = 0, 2 | exactly 0 | selection rule |, |

f(1s→2p) and the 2p lifetime are the tight anchors (confident reference
values). Others are checked structurally (positivity, monotonicity, the
selection-rule zeros) rather than against digits I cannot verify.

## Deferred

Server endpoint and Spectrum-view intensity rendering (bar heights / opacity by
A or f), and the fine-structure line strengths (j-resolved). The engine here is
the seam those will call.
