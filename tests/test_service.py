from fastapi.testclient import TestClient

from preloved.search.stack import Design, Stack
from preloved.service.app import create_app


def client(world, settings):
    st = Stack(settings)
    st.add_listings([it.listing() for it in world.items])
    designs = {
        "lexical": Design("lexical"),
        "understood": Design("understood", understand=True, index="structured", hard=("brand", "category")),
    }
    return TestClient(create_app(st, designs, world.available_mask(0), 0, {"world": "test"}))


def test_search_and_designs(small_world, settings):
    c = client(small_world, settings)
    assert c.get("/health").json()["ok"]
    assert {d["name"] for d in c.get("/designs").json()} == {"lexical", "understood"}
    r = c.get("/search", params={"q": "sac chanel noir", "design": "understood"}).json()
    assert r["understood"]["brand"] == "Chanel" and len(r["items"]) <= settings["page_size"]
    assert all("appeal" not in x for x in r["items"])
    assert c.get("/report").json() == {"world": "test"}


def test_errors_say_what_to_do(small_world, settings):
    c = client(small_world, settings)
    r = c.get("/search", params={"q": "bag", "design": "nope"})
    assert r.status_code == 404 and "/designs" in r.json()["detail"]
    assert c.get("/search", params={"q": ""}).status_code == 422
