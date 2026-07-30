from math import comb

import pytest

from atomsim.atoms import (
    ATOM_KEYS,
    atom_for_key,
    aufbau_configuration,
    element_by_symbol,
    format_config,
    is_atom_key,
    is_ground,
    is_single_term,
    open_subshells,
    parse_config,
    subshell_capacity,
    subshell_term_count,
    subshell_terms,
    total_electrons,
    validate_config,
)


def test_subshell_capacity():
    assert subshell_capacity(0) == 2   # s
    assert subshell_capacity(1) == 6   # p
    assert subshell_capacity(2) == 10  # d


@pytest.mark.parametrize("z,expected", [
    (1, "1s1"),
    (2, "1s2"),
    (6, "1s2 2s2 2p2"),      # carbon
    (10, "1s2 2s2 2p6"),     # neon
    (11, "1s2 2s2 2p6 3s1"), # sodium
    (18, "1s2 2s2 2p6 3s2 3p6"),  # argon
])
def test_aufbau_matches_known_configs(z, expected):
    assert format_config(aufbau_configuration(z)) == expected


def test_config_roundtrip_and_count():
    cfg = parse_config("1s2 2s2 2p1")
    assert total_electrons(cfg) == 5
    assert format_config(cfg) == "1s2 2s2 2p1"


def test_is_ground():
    assert is_ground(aufbau_configuration(11)) is True
    assert is_ground(parse_config("1s2 2s2 2p6 3p1")) is False  # excited Na


def test_validate_rejects_overfill_and_bad_shell():
    with pytest.raises(ValueError, match="capacity"):
        validate_config(parse_config("1s3"))         # > 2 in s
    with pytest.raises(ValueError, match="n must be"):
        validate_config(((( 1, 1), 1),))              # 1p impossible (n<=l)


@pytest.mark.parametrize("l,q,terms", [
    (0, 1, 1),   # s1  -> 2S
    (0, 2, 1),   # s2  -> 1S
    (1, 1, 1),   # p1  -> 2P
    (1, 2, 3),   # p2  -> 3P, 1D, 1S
    (1, 3, 3),   # p3  -> 4S, 2D, 2P
    (1, 4, 3),   # p4  -> same as p2, by particle-hole symmetry
    (1, 5, 1),   # p5  -> 2P, one hole
    (1, 6, 1),   # p6  -> 1S
    (2, 1, 1),   # d1  -> 2D
    (2, 2, 5),   # d2  -> 3F, 3P, 1G, 1D, 1S
    (2, 3, 8),   # d3
    (2, 5, 16),  # d5, the half-filled worst case
])
def test_subshell_term_counts_match_the_textbook_tables(l, q, terms):
    """Standard Russell-Saunders term counts for equivalent electrons, e.g.
    Condon & Shortley Table 1^3; the p and d columns are the ones every atomic
    structure text prints."""
    assert subshell_term_count(l, q) == terms


@pytest.mark.parametrize(
    "l,q",
    [(1, q) for q in range(7)] + [(2, q) for q in range(11)] + [(3, 3), (3, 7)],
)
def test_terms_account_for_every_determinant(l, q):
    """Independent check on the peel, which the bare count cannot make: the
    degeneracies of the terms found must sum to the number of ways q electrons
    fit in 2(2l+1) spin-orbitals. A dropped or double-struck rectangle changes
    this sum even when it leaves the count intact."""
    degeneracy = sum(
        (2 * big_l + 1) * (twice_s + 1) for big_l, twice_s in subshell_terms(l, q)
    )
    assert degeneracy == comb(subshell_capacity(l), q)


def test_terms_name_the_expected_states_for_p2():
    """3P, 1D, 1S as (L, 2S) pairs, in the order the peel finds them."""
    assert subshell_terms(1, 2) == ((1, 2), (2, 0), (0, 0))


def test_term_enumeration_rejects_an_impossible_occupancy():
    with pytest.raises(ValueError, match="out of range"):
        subshell_terms(1, 7)


@pytest.mark.parametrize("z,single", [
    (2, True),    # He 1s2, closed
    (3, True),    # Li 2s1  -> 2S alone
    (5, True),    # B  2p1  -> 2P alone
    (6, False),   # C  2p2  -> 3P, 1D, 1S
    (7, False),   # N  2p3
    (8, False),   # O  2p4
    (9, True),    # F  2p5  -> 2P alone
    (10, True),   # Ne closed
    (16, False),  # S  3p4
    (17, True),   # Cl 3p5
    (18, True),   # Ar closed
])
def test_single_term_configurations(z, single):
    assert is_single_term(aufbau_configuration(z)) is single


def test_two_open_subshells_never_span_a_single_term():
    """Excited carbon, 2s1 2p3: two open subshells couple to many terms, and the
    count is short-circuited rather than enumerated."""
    cfg = parse_config("1s2 2s1 2p3")
    assert len(open_subshells(cfg)) == 2
    assert is_single_term(cfg) is False


def test_open_subshells_ignores_full_ones():
    assert open_subshells(aufbau_configuration(10)) == ()
    assert open_subshells(aufbau_configuration(6)) == (((2, 1), 2),)


def test_atom_keys_cover_he_to_ar_minus_s_cl():
    # Presets span He..Ar but omit S and Cl, which have no published neutral GSZ
    # parameters (Szydlik & Green 1974, Table I) — 15 atoms, not 17.
    assert ATOM_KEYS[0] == "he" and ATOM_KEYS[-1] == "ar"
    assert len(ATOM_KEYS) == 15
    assert is_atom_key("na") and not is_atom_key("h")
    assert not is_atom_key("s") and not is_atom_key("cl")
    assert atom_for_key("na").z == 11 and element_by_symbol("Na").z == 11
