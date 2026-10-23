import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import _with_state_preview, create_app
from atomsim.server.thumbnails import OG_HEIGHT, OG_WIDTH, render_og_card, render_thumbnail

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def test_render_thumbnail_returns_png_bytes():
    render_thumbnail.cache_clear()
    png = render_thumbnail(2, 1, 0, "h", "complex", 64)
    assert isinstance(png, bytes)
    assert png[:8] == PNG_MAGIC


def test_render_thumbnail_caches():
    render_thumbnail.cache_clear()
    render_thumbnail(1, 0, 0, "h", "complex", 48)
    render_thumbnail(1, 0, 0, "h", "complex", 48)
    assert render_thumbnail.cache_info().hits == 1


def test_thumbnail_endpoint(client):
    r = client.get("/api/thumbnail/2/1/0?size=48")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert "max-age" in r.headers["cache-control"]
    assert r.content[:8] == PNG_MAGIC


def test_thumbnail_states_differ(client):
    a = client.get("/api/thumbnail/1/0/0?size=48").content
    b = client.get("/api/thumbnail/2/1/0?size=48").content
    assert a != b


def test_thumbnail_validation(client):
    assert client.get("/api/thumbnail/2/1/0?size=10").status_code == 422
    assert client.get("/api/thumbnail/2/1/0?size=999").status_code == 422
    assert client.get("/api/thumbnail/2/1/0?basis=cartoon").status_code == 422
    assert client.get("/api/thumbnail/1/1/0").status_code == 422
    assert client.get("/api/thumbnail/2/1/0?system=unobtainium").status_code == 422


def _png_size(png: bytes) -> tuple[int, int]:
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def test_og_card_is_the_open_graph_size(client):
    r = client.get("/api/og/3/1/-1?system=mu-h")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert _png_size(r.content) == (OG_WIDTH, OG_HEIGHT)


def test_og_card_states_differ():
    assert render_og_card(2, 1, 0, "h", "real") != render_og_card(2, 1, 1, "h", "real")


def test_og_card_refuses_what_it_cannot_draw(client):
    assert client.get("/api/og/7/0/0").status_code == 422  # past the app's n select
    assert client.get("/api/og/2/1/2").status_code == 422
    assert client.get("/api/og/2/1/0?system=na").status_code == 422  # not closed-form
    assert client.get("/api/og/2/1/0?basis=cartoon").status_code == 422


TEMPLATE = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text("utf-8")


def _meta(html: str, key: str) -> str:
    return re.search(rf'(?:property|name)="{key}"\s+content="([^"]*)"', html).group(1)


def test_deep_link_points_its_preview_at_its_own_state():
    q = {"n": "3", "l": "1", "m": "-1", "system": "mu-h", "view": "plane"}
    html = _with_state_preview(TEMPLATE, q)
    image = "%VITE_SITE_URL%/api/og/3/1/-1?system=mu-h&amp;basis=complex"
    assert _meta(html, "og:image") == image
    assert _meta(html, "twitter:image") == image
    assert _meta(html, "og:title") == "Muonic hydrogen 3p, m = -1 | AtomSim"
    assert "3p, m = -1" in _meta(html, "og:image:alt")  # the multi-line tag matched too
    assert (_meta(html, "og:image:width"), _meta(html, "og:image:height")) == ("1200", "630")


def test_front_page_and_undrawable_links_keep_the_built_preview():
    assert _with_state_preview(TEMPLATE, {}) == TEMPLATE
    assert _with_state_preview(TEMPLATE, {"view": "levels"}) == TEMPLATE
    assert _with_state_preview(TEMPLATE, {"system": "ar"}) == TEMPLATE
    assert _with_state_preview(TEMPLATE, {"n": "9"}) == TEMPLATE
    assert _with_state_preview(TEMPLATE, {"n": "2", "l": "1", "basis": '"><script>'}) == TEMPLATE
