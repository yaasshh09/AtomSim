"""The gallery thumbnails: small inferno PNGs of plane densities.

They are navigation aids, not measurement surfaces: the brightness is
gamma-compressed (t = (rho/rho_max)^GAMMA) so faint outer lobes stay visible, a
VISUAL LIBERTY disclosed in the gallery UI. The frontend cross-section renderer
mirrors this GAMMA and the matplotlib LUT (web/src/lib/colormap.ts, luts.ts), so
a density looks the same everywhere.
"""

import io
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")

from matplotlib import image as mpl_image  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from atomsim.analytic.angular import real_orbital_label  # noqa: E402
from atomsim.plane import plane_grid  # noqa: E402
from atomsim.spectra import orbital_label  # noqa: E402
from atomsim.systems import get_system  # noqa: E402

GAMMA = 0.5


@lru_cache(maxsize=512)
def render_thumbnail(n: int, l: int, m: int, system: str, basis: str, size: int) -> bytes:
    """An inferno PNG of |psi|^2 on y=0, with the row order flipped so +z is up."""
    sys_ = get_system(system)
    pg = plane_grid(
        n, l, m, quantity="density", basis=basis,
        Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, resolution=size,
    )
    rho = pg.values
    vmax = float(rho.max())
    t = (rho / vmax) ** GAMMA if vmax > 0.0 else rho
    buf = io.BytesIO()
    mpl_image.imsave(buf, t[::-1], cmap="inferno", vmin=0.0, vmax=1.0, format="png")
    return buf.getvalue()


#: Open Graph's recommended card size; every major scraper crops to 1.91:1.
OG_WIDTH, OG_HEIGHT = 1200, 630


def state_title(n: int, l: int, m: int, basis: str) -> str:
    """How a shared link names its state: "3p, m = -1" or "3p_x"."""
    if basis == "real":
        return f"{n}{real_orbital_label(l, m)}"
    return f"{orbital_label(n, l)}, m = {m}"


@lru_cache(maxsize=128)
def render_og_card(n: int, l: int, m: int, system: str, basis: str) -> bytes:
    """The link-preview card for one state: its real y=0 density plus what it is.

    The picture is the same gamma-compressed inferno density as the thumbnail,
    so the card says so; the density itself is the analytic one (EXACT).
    """
    sys_ = get_system(system)
    pg = plane_grid(
        n, l, m, quantity="density", basis=basis,
        Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, resolution=OG_HEIGHT,
    )
    rho = pg.values
    vmax = float(rho.max())
    t = (rho / vmax) ** GAMMA if vmax > 0.0 else rho

    dpi = 100
    fig = Figure(figsize=(OG_WIDTH / dpi, OG_HEIGHT / dpi), dpi=dpi, facecolor="#0b0b0c")
    img_ax = fig.add_axes((1 - OG_HEIGHT / OG_WIDTH, 0, OG_HEIGHT / OG_WIDTH, 1))
    img_ax.imshow(t, cmap="inferno", vmin=0.0, vmax=1.0, origin="lower",
                  interpolation="bilinear")
    img_ax.set_axis_off()

    ink, dim = "#f2f2f2", "#9a9a9f"
    fig.text(0.04, 0.84, "AtomSim", color=dim, fontsize=22)
    title = state_title(n, l, m, basis)
    # The text column is 570 px; "6f_x(x2-3y2)" at the full size runs into the image.
    fig.text(0.04, 0.62, title, color=ink, fontsize=58 if len(title) <= 10 else 40,
             fontweight="bold")
    fig.text(0.04, 0.52, sys_.name, color=ink, fontsize=30)
    fig.text(0.04, 0.34, "|ψ|² on the y = 0 plane, +z up", color=dim, fontsize=20)
    fig.text(0.04, 0.27, "density: EXACT (closed-form hydrogen-like)", color=dim, fontsize=16)
    fig.text(0.04, 0.21, f"brightness: (ρ/ρmax)^{GAMMA}, VISUAL LIBERTY",
             color=dim, fontsize=16)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return buf.getvalue()
