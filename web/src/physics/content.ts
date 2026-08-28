import type { ViewMode } from "../state/store";

export interface PhysicsBlock {
  tex: string;
  note: string;
}

export const PHYSICS_CONTENT: Record<
  ViewMode,
  { title: string; blocks: PhysicsBlock[] }
> = {
  cloud: {
    title: "What the point cloud is",
    blocks: [
      {
        tex: String.raw`\psi_{n\ell m}(r,\theta,\varphi) = R_{n\ell}(r)\,Y_\ell^m(\theta,\varphi)`,
        note: "The stationary state factorizes into the closed-form radial part and a spherical harmonic. I compute both in the engine and never approximate either in your browser.",
      },
      {
        tex: String.raw`p(\mathbf r)\,dV = |\psi_{n\ell m}(\mathbf r)|^2\,dV`,
        note: "Each dot is one independent draw I took from |ψ|² (seeded inverse-CDF Monte-Carlo). What I am showing you is a histogram of position measurements, not a photograph of an object.",
      },
    ],
  },
  plane: {
    title: "The cross-section, honestly",
    blocks: [
      {
        tex: String.raw`\rho(x, 0, z) = |\psi_{n\ell m}(x, 0, z)|^2`,
        note: "Probability density on the plane containing the quantization axis. The classic poster labels a signed quantity 'probability density', and a density is non-negative, so I label ψ and |ψ|² separately.",
      },
      {
        tex: String.raw`e^{im\varphi}\big|_{y=0} = \pm 1 \;\Rightarrow\; \psi\big|_{y=0} \in \mathbb{R}`,
        note: "On y = 0 the azimuthal factor is ±1, so ψ itself is real there: my signed-ψ view is exact on this plane, not a convention I adopted.",
      },
    ],
  },
  radial: {
    title: "Radial structure",
    blocks: [
      {
        tex: String.raw`P_{n\ell}(r) = r^2\,|R_{n\ell}(r)|^2,\qquad \int_0^\infty P_{n\ell}(r)\,dr = 1`,
        note: "P(r) is the probability density for the electron's distance from the nucleus; the r² factor is the volume of the spherical shell.",
      },
      {
        tex: String.raw`\langle r\rangle = \frac{a_0\,m_e}{Z\,\mu}\;\frac{3n^2 - \ell(\ell+1)}{2}`,
        note: "The marker I dash in is the quantum expectation value, not the Bohr-model radius n²a₀ that many visualizers quietly show instead.",
      },
    ],
  },
  levels: {
    title: "Level energies",
    blocks: [
      {
        tex: String.raw`E_n = -\frac{Z^2}{2n^2}\,\frac{\mu}{m_e}\,E_h`,
        note: "I am exact in the reduced mass here (EXACT badge): isotope and exotic-system dependence reaches me only through μ.",
      },
      {
        tex: String.raw`\Delta E_{nj} = -\frac{(Z\alpha)^2\,|E_n|}{n}\left(\frac{1}{j+\tfrac12} - \frac{3}{4n}\right)`,
        note: "The α² fine structure (spin-orbit + relativistic kinetic energy + Darwin term, combined). APPROXIMATION badge: I leave out the α⁴ terms and the Lamb shift, which is why equal-j levels coincide for me.",
      },
    ],
  },
  whatif: {
    title: "What α does (and does not) touch",
    blocks: [
      {
        tex: String.raw`E_n = -\frac{Z^2}{2n^2}\,\frac{\mu}{m_e}\,E_h`,
        note: "The gross ladder in Hartree atomic units contains no α, so when you turn the slider I hold these rungs fixed (EXACT). Z enters as Z².",
      },
      {
        tex: String.raw`\Delta E_{nj} = -\frac{(Z\alpha)^2\,|E_n|}{n}\left(\frac{1}{j+\tfrac12} - \frac{3}{4n}\right)`,
        note: "The fine split scales as (Zα)², and this is the term my lab drives. APPROXIMATION badge: my perturbative error itself grows as (Zα)², and past the validity limit I disclose, the exact Dirac result would differ.",
      },
    ],
  },
  spectrum: {
    title: "Where the lines come from",
    blocks: [
      {
        tex: String.raw`\frac{1}{\lambda} = R_M Z^2\left(\frac{1}{n_1^2} - \frac{1}{n_2^2}\right),\qquad R_M = \frac{\mu}{m_e}\,R_\infty`,
        note: "I build lines from level differences and filter them by the selection rules Δl = ±1 (and Δj = 0, ±1 with fine structure on). I compare my wavelengths against vendored NIST data at a tolerance I state.",
      },
    ],
  },
  forcelaw: {
    title: "What if Coulomb weren't 1/r?",
    blocks: [
      {
        tex: String.raw`V_p(r) = -\frac{Z}{r^{\,p}},\qquad 0.5 \le p \le 1.5`,
        note: "A counterfactual central potential with no closed form away from p = 1, which I solve on a radial grid (NUMERICAL badge: I give each level a grid-halving error estimate). I keep p well below the fall-to-center threshold at p → 2.",
      },
      {
        tex: String.raw`p = 1 \;\Rightarrow\; E_{n\ell} = -\frac{Z^2}{2n^2}\,\frac{\mu}{m_e}\,E_h \quad\text{for every } \ell`,
        note: "Only the exact 1/r Coulomb shape hides l from the energy (the accidental degeneracy). Bend p away from 1 and it lifts, so my counterfactual ladder splits against the EXACT hydrogen reference I draw beside it.",
      },
    ],
  },
};
