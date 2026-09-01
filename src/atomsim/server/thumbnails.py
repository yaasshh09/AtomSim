"""The gallery thumbnails I render myself: small inferno PNGs of plane densities.

They are navigation aids, not measurement surfaces: I gamma-compress the
brightness (t = (rho/rho_max)^GAMMA) so faint outer lobes stay visible, a VISUAL
LIBERTY I disclose in the gallery UI. The frontend cross-section renderer
mirrors my GAMMA and my matplotlib LUT (web/src/lib/colormap.ts, luts.ts), so a
density looks the same everywhere.
"""

import io
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")

from matplotlib import image as mpl_image  # noqa: E402

from atomsim.plane import plane_grid  # noqa: E402
from atomsim.systems import get_system  # noqa: E402

GAMMA = 0.5


@lru_cache(maxsize=512)
def render_thumbnail(n: int, l: int, m: int, system: str, basis: str, size: int) -> bytes:
    """My inferno PNG of |psi|^2 on y=0; I flip the row order so +z is up."""
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
