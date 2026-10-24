"""Every colour map the app ships, checked under simulated colour blindness.

Simulation: the severity-100 matrices of Machado, Oliveira & Fernandes, "A
Physiologically-based Model for Simulation of Color Vision Deficiency", IEEE
TVCG 15, 1291 (2009), doi:10.1109/TVCG.2009.113, applied in linear sRGB. The
numbers are the paper's supplementary tables as vendored by colorspacious
(colorspacious/cvd.py, checked against it 2026-10-24). Distances are CIELAB
dE76 under D65.

What each map has to keep, for every reader:
  inferno (density): lightness that rises monotonically, so brighter means more.
  RdBu_r (signed):   its two arms apart, so a sign is never mistaken.
  PHASE (arg psi):   any two phases a sixth of a turn apart still distinct, and
                     no phase lost against the white or the black stage.
"""

import colorsys
import re
from pathlib import Path

import numpy as np
import pytest

LUTS = (Path(__file__).resolve().parents[1] / "web" / "src" / "lib" / "luts.ts").read_text("utf-8")

MACHADO = {
    "protanopia": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                   [-0.003882, -0.048116, 1.051998]],
    "deuteranopia": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                     [-0.011820, 0.042940, 0.968881]],
    "tritanopia": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                   [0.004733, 0.691367, 0.303900]],
    "typical": np.eye(3).tolist(),
}


def _lut(name: str) -> np.ndarray:
    block = re.search(rf"export const {name}: Lut = \[(.*?)\];", LUTS, re.S).group(1)
    rows = re.findall(r"\[(\d+), (\d+), (\d+)\]", block)
    return np.array(rows, dtype=float) / 255.0


def _linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(rgb, cvd):
    lin = np.clip(_linear(rgb) @ np.array(MACHADO[cvd]).T, 0, 1)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                          [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _min_cyclic_distance(lab: np.ndarray, min_steps: int) -> float:
    n = len(lab)
    worst = np.inf
    for shift in range(min_steps, n - min_steps + 1):
        worst = min(worst, np.linalg.norm(lab - np.roll(lab, shift, axis=0), axis=1).min())
    return worst


@pytest.mark.parametrize("cvd", MACHADO)
def test_inferno_keeps_rising_lightness(cvd):
    assert np.all(np.diff(_lab(_lut("INFERNO"), cvd)[:, 0]) > 0)


@pytest.mark.parametrize("cvd", MACHADO)
def test_rdbu_keeps_its_two_signs_apart(cvd):
    lab = _lab(_lut("RDBU_R"), cvd)
    mirrored = [np.linalg.norm(lab[i] - lab[255 - i]) for i in range(115)]  # |t - 1/2| > 0.05
    assert min(mirrored) > 8


@pytest.mark.parametrize("cvd", MACHADO)
def test_phase_keeps_a_sixth_of_a_turn_distinct(cvd):
    assert _min_cyclic_distance(_lab(_lut("PHASE"), cvd), 256 // 6) > 15


def test_no_phase_disappears_into_either_stage():
    L = _lab(_lut("PHASE"), "typical")[:, 0]
    assert L.min() > 30 and L.max() < 90  # dE > 10 in lightness from black (0) and white (100)


def test_the_old_hue_wheel_is_why():
    """The HSL wheel this replaced: a sixth of a turn apart, indistinguishable for deuteranopes."""
    wheel = np.array([colorsys.hls_to_rgb(i / 256, 0.55, 1.0) for i in range(256)])
    assert _min_cyclic_distance(_lab(wheel, "deuteranopia"), 256 // 6) < 1
