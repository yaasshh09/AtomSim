"""A hydrogen-like atom driven by a laser: Rabi oscillations with real decay.

A continuous, linearly polarized (z) laser near resonance with one dipole
transition couples the two sublevels |g> = |n l m> and |e> = |n' l' m> (pi light
keeps m). The coupling strength is the Rabi frequency

    Omega = <e| e z |g> E0 / hbar,   E0 = sqrt(2 I / (c eps0))

built from the engine's exact radial dipole integral and the closed-form
angular factor. Without decay the excited population oscillates exactly as

    P_e(t) = (Omega^2 / Omega_eff^2) sin^2(Omega_eff t / 2),  Omega_eff^2 = Omega^2 + Delta^2

With decay, the excited sublevel empties at its full rate Gamma = 1 / tau, but
only the pi part of its decay to the driven lower sublevel comes back; the rest
(other levels, other m) leaves the driven pair for good and is tracked as "lost".
That is the Lindblad master equation for three states (g, e, lost), solved by a
matrix exponential of the Liouvillian.

APPROXIMATION: two driven sublevels, the rotating-wave approximation (RWA),
electric-dipole coupling, a monochromatic field with no Doppler or ionization.
The RWA drops terms of relative size Omega / omega_0; past 0.1 it is refused.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import constants as _sc
from scipy.linalg import expm

from atomsim.analytic.hydrogen import energy, validate_quantum_numbers
from atomsim.analytic.transitions import dipole_radial_integral, einstein_A, lifetime
from atomsim.provenance import Fidelity, Field, Provenance, Quantity

_T_AU = _sc.physical_constants["atomic unit of time"][0]
_E_AU = _sc.physical_constants["atomic unit of electric field"][0]  # V/m

#: Past this Omega / omega_0 the counter-rotating terms are not small.
RWA_LIMIT = 0.1


@dataclass(frozen=True)
class RabiDrive:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]
    dipole: Quantity             # <e|z|g> in bohr
    rabi_frequency: Quantity     # Omega, rad/s
    effective_frequency: Quantity  # Omega_eff, rad/s
    decay_rate: Quantity         # Gamma of the excited sublevel, s^-1
    return_fraction: Quantity    # share of Gamma that comes back to |g>
    rwa_parameter: Quantity      # Omega / omega_0
    steady_excited: Quantity | None  # rho_ee as t -> inf; None when population leaks
    excited: Field               # P_e(t) with decay
    excited_undamped: Field      # the closed form, no decay
    lost: Field                  # population that left the driven pair


def angular_factor_z(l: int, l2: int, m: int) -> float:
    """<l2 m| cos(theta) |l m>, nonzero only for l2 = l +/- 1."""
    if abs(l2 - l) != 1 or abs(m) > min(l, l2):
        return 0.0
    lo = min(l, l2)
    return math.sqrt(((lo + 1) ** 2 - m * m) / ((2 * lo + 1) * (2 * lo + 3)))


def _liouvillian(omega: float, delta: float, a_back: float, a_lost: float) -> np.ndarray:
    """d vec(rho)/dt = L vec(rho) for states (g, e, lost), row-major vec."""
    H = np.zeros((3, 3), dtype=complex)
    H[1, 1] = -delta
    H[0, 1] = H[1, 0] = omega / 2.0
    eye = np.eye(3)
    L = -1j * (np.kron(H, eye) - np.kron(eye, H.T))
    for rate, target in ((a_back, 0), (a_lost, 2)):
        if rate <= 0.0:
            continue
        C = np.zeros((3, 3))
        C[target, 1] = math.sqrt(rate)
        CdC = C.T @ C
        L += np.kron(C, C) - 0.5 * (np.kron(CdC, eye) + np.kron(eye, CdC.T))
    return L


def rabi_drive(
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    intensity_w_m2: float,
    detuning_mhz: float = 0.0,
    Z: int = 1,
    mu_ratio: float = 1.0,
    decay: bool = True,
    points: int = 600,
) -> RabiDrive:
    """Drive |lower> <-> |upper> (both (n, l, m), same m) with a z-polarized laser."""
    (n, l, m), (n2, l2, m2) = lower, upper
    validate_quantum_numbers(n, l)
    validate_quantum_numbers(n2, l2)
    if m != m2:
        raise ValueError("z-polarized (pi) light keeps m: the two sublevels need the same m")
    ang = angular_factor_z(l, l2, m)
    if ang == 0.0:
        raise ValueError(
            f"no pi coupling between l={l} and l={l2} at m={m} (needs l' = l +/- 1, |m| <= both)"
        )
    e_low = energy(n, Z=Z, mu_ratio=mu_ratio).value
    e_up = energy(n2, Z=Z, mu_ratio=mu_ratio).value
    if e_up <= e_low:
        raise ValueError("the upper level must lie above the lower one")
    if not intensity_w_m2 > 0.0:
        raise ValueError("the intensity must be positive")

    R = dipole_radial_integral(n, l, n2, l2, Z=Z, mu_ratio=mu_ratio)
    d = R.value * ang
    e0_au = math.sqrt(2.0 * intensity_w_m2 / (_sc.c * _sc.epsilon_0)) / _E_AU
    omega = abs(d) * e0_au / _T_AU                     # rad/s
    omega0 = (e_up - e_low) / _T_AU                    # rad/s
    rwa = omega / omega0
    if rwa > RWA_LIMIT:
        raise ValueError(
            f"Omega / omega_0 = {rwa:.2g} exceeds {RWA_LIMIT}: the rotating-wave "
            "approximation fails, so this intensity is refused rather than solved wrongly"
        )
    delta = 2.0 * math.pi * detuning_mhz * 1e6
    omega_eff = math.hypot(omega, delta)

    gamma = 1.0 / lifetime(n2, l2, Z=Z, mu_ratio=mu_ratio).value if decay else 0.0
    # The pi share of the decay to the driven lower sublevel: the summed angular
    # factor over all m', q is l_max / (2l'+1), and this one channel is ang^2.
    a_pair = einstein_A(n2, l2, n, l, Z=Z, mu_ratio=mu_ratio).value
    a_back = a_pair * ang**2 * (2 * l2 + 1) / max(l, l2) if decay else 0.0
    back_fraction = a_back / gamma if gamma > 0.0 else 1.0

    periods = 2.0 * math.pi / omega_eff
    t_max = 8.0 * periods
    if gamma > 0.0:
        t_max = max(t_max, min(5.0 / gamma, 40.0 * periods))
    t = np.linspace(0.0, t_max, points)

    L = _liouvillian(omega, delta, a_back, gamma - a_back)
    step = expm(L * (t[1] - t[0]))
    rho = np.zeros(9, dtype=complex)
    rho[0] = 1.0                                       # all in |g>
    pe = np.empty(points)
    pl = np.empty(points)
    for i in range(points):
        pe[i] = rho[4].real
        pl[i] = rho[8].real
        rho = step @ rho
    closed = (omega / omega_eff) ** 2 * np.sin(omega_eff * t / 2.0) ** 2

    assumptions = (
        "two sublevels driven by pi light; every other sublevel is outside the model",
        f"rotating-wave approximation: Omega / omega_0 = {rwa:.2g}, the relative "
        "size of the dropped counter-rotating terms (Bloch-Siegert shift ~ Omega^2 / 4 omega_0)",
        "electric-dipole coupling to a monochromatic continuous wave; no Doppler, "
        "no photoionization, no laser linewidth",
        "decay out of the pair (other levels, other m) is counted as lost, not re-pumped",
    )
    prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method="Lindblad master equation for (g, e, lost), propagated by expm(L dt)",
        assumptions=assumptions,
        error_estimate=rwa,
        refinement="Floquet treatment without the RWA; the full sublevel manifold",
    )
    hz = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method="Omega = |<e|z|g>| E0 / hbar, E0 = sqrt(2 I / (c eps0))",
        assumptions=assumptions,
        error_estimate=omega * (R.provenance.error_estimate or 0.0) / abs(R.value),
    )
    steady = None
    if decay and back_fraction > 1.0 - 1e-12:
        s = 2.0 * omega**2 / gamma**2
        rho_ee = 0.5 * s / (1.0 + s + (2.0 * delta / gamma) ** 2)
        steady = Quantity(rho_ee, "dimensionless", "rho_ee steady state", Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="(s/2) / (1 + s + (2 Delta/Gamma)^2), s = 2 Omega^2 / Gamma^2",
            assumptions=assumptions,
            error_estimate=rwa * rho_ee,
        ))
    label_pair = f"{n}{l}{m}->{n2}{l2}{m2}"
    return RabiDrive(
        lower=lower,
        upper=upper,
        dipole=Quantity(d, "bohr", f"<e|z|g> {label_pair}", Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="radial dipole integral times <l' m|cos theta|l m> (closed form)",
            assumptions=R.provenance.assumptions,
            error_estimate=abs(ang) * (R.provenance.error_estimate or 0.0),
        )),
        rabi_frequency=Quantity(omega, "rad/s", "Omega", hz),
        effective_frequency=Quantity(omega_eff, "rad/s", "Omega_eff", hz),
        decay_rate=Quantity(gamma, "s^-1", "Gamma", Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="Gamma = 1 / tau of the upper level, all E1 channels" if decay
            else "decay switched off",
            assumptions=R.provenance.assumptions,
        )),
        return_fraction=Quantity(back_fraction, "dimensionless", "returns to |g>", Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="A(pair) ang^2 (2l'+1) / l_max, over Gamma",
            assumptions=R.provenance.assumptions,
        )),
        rwa_parameter=Quantity(rwa, "dimensionless", "Omega / omega_0", Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="Rabi frequency over the transition frequency",
            assumptions=assumptions,
        )),
        steady_excited=steady,
        excited=Field(pe, t, "dimensionless", "s", "P_e(t)", prov),
        excited_undamped=Field(closed, t, "dimensionless", "s", "P_e(t), no decay", Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="(Omega/Omega_eff)^2 sin^2(Omega_eff t / 2), closed form",
            assumptions=assumptions,
            error_estimate=rwa,
        )),
        lost=Field(pl, t, "dimensionless", "s", "P_lost(t)", prov),
    )
