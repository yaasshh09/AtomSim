"""Suite-wide fixtures.

The job-endpoint rate limiter is on by default (see `_build_rate_limiter`),
which is right for a deployed server and wrong for a test suite: the suite
fires job POSTs far faster than any human, from one apparent client, and would
throttle itself into failures that say nothing about physics.

Switching it off here rather than raising its ceiling keeps the intent legible.
The limiter's own behaviour is covered directly in `test_ratelimit.py`, and its
wiring into the app in `test_server_ratelimit.py`, both of which build their
own configuration instead of relying on this default.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_rate_limit_for_the_suite():
    previous = os.environ.get("ATOMSIM_RATE_LIMIT")
    os.environ["ATOMSIM_RATE_LIMIT"] = "off"
    yield
    if previous is None:
        del os.environ["ATOMSIM_RATE_LIMIT"]
    else:
        os.environ["ATOMSIM_RATE_LIMIT"] = previous
