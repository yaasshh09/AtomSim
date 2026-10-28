"""The teacher worksheets quote numbers; each one is checked here against the
engine (tour-claim resolvers) or the vendored NIST data, so an answer key
cannot drift away from what the app shows."""

import json
import re
from pathlib import Path

import pytest

from atomsim.hf_reference import nist_ionization_energy
from atomsim.tour_claims import resolve_claim

PAGE = (Path(__file__).resolve().parents[1] / "web" / "worksheets.html").read_text(
    encoding="utf-8"
)
CLAIMS = re.findall(r"data-claim='(\{[^']+\})'>([^<]+)<", PAGE)
MEASURED = re.findall(r'data-nist-ie="(\d+)">([^<]+)<', PAGE)


def _number(text: str) -> float:
    return float(text.replace("−", "-").split()[0])


def test_the_page_has_claims_to_check():
    assert len(CLAIMS) >= 10
    assert len(MEASURED) >= 2


@pytest.mark.parametrize("raw,text", CLAIMS)
def test_every_quoted_value_is_the_engines(raw, text):
    claim = json.loads(raw)
    assert abs(resolve_claim(claim) - _number(text)) <= claim["tol"], (claim, text)


@pytest.mark.parametrize("z,text", MEASURED)
def test_every_measured_value_is_nists(z, text):
    assert _number(text) == pytest.approx(nist_ionization_energy(int(z))["ie_ev"], abs=0.006)
