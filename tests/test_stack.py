import numpy as np

from preloved.search.ranker import FEATURES, Ranker
from preloved.search.stack import Design, Stack
from preloved.search.vectors import BUCKETS, TwoTower


def build(world, settings, towers=None, rankers=None):
    st = Stack(settings, towers=towers, rankers=rankers)
    st.add_listings([it.listing() for it in world.items])
    return st


def test_only_items_for_sale_are_returned(small_world, settings):
    st = build(small_world, settings)
    avail = small_world.available_mask(0)
    avail[::2] = False
    for s in small_world.searches_for_day(0)[:30]:
        r = st.search(Design("lexical"), s.query, s.user_id, s.market, 0, avail)
        assert all(avail[i] for i in r.items) and len(r.items) <= settings["page_size"]


def test_hard_filters_hold_and_relax_when_too_narrow(small_world, settings):
    st = build(small_world, settings)
    avail = small_world.available_mask(0)
    d = Design("understood", understand=True, index="structured", hard=("brand", "category"))
    r = st.search(d, "chanel bag", 0, "FR", 0, avail)
    assert r.items and not r.relaxed
    assert all(
        st.cat.listings[i]["brand"] in ("Chanel", None) or st.cat.a_slots["brand"][i] == "Chanel" for i in r.items
    )
    few = np.zeros_like(avail)
    chanel = np.flatnonzero(st.cat.a_slots["brand"] == "Chanel")[:3]
    few[chanel] = True
    few[np.flatnonzero(st.cat.a_slots["brand"] != "Chanel")[:200]] = True
    r = st.search(d, "chanel bag", 0, "FR", 0, few)
    assert r.relaxed  # three Chanel items are not a page: the narrowest filter is dropped
    strict = Design("strict", understand=True, index="structured", hard=("brand", "category"), relax=False)
    assert len(st.search(strict, "chanel bag", 0, "FR", 0, few).items) <= 3


def test_vectors_and_ranker_compose(small_world, settings):
    r = np.random.default_rng(0)
    tower = TwoTower(r.normal(size=(BUCKETS, 8)), r.normal(size=(BUCKETS, 8)), {})
    h = 4
    w = {
        "w1": r.normal(size=(h, len(FEATURES))),
        "b1": np.zeros(h),
        "w2": r.normal(size=(1, h)),
        "b2": np.zeros(1),
        "mean": np.zeros(len(FEATURES)),
        "std": np.ones(len(FEATURES)),
    }
    st = build(small_world, settings, towers={"content": tower}, rankers={"ips": Ranker(w, {})})
    assert st.vecs["content"].shape == (len(small_world.items), 8)
    d = Design(
        "personalised",
        understand=True,
        index="structured",
        hard=("brand", "category"),
        vectors="content",
        fusion="rrf",
        ranker="ips",
        depth=50,
    )
    avail = small_world.available_mask(0)
    res = st.search(d, "sac chanel noir", 1, "FR", 0, avail)
    assert len(res.items) == settings["page_size"] and len(set(res.items)) == len(res.items)


def test_profiles_come_from_the_click_log(small_world, settings):
    st = build(small_world, settings)
    st.observe_clicks([(5, 0), (5, 1)])
    assert st.profiles[5].clicks == 2 and len(st.profiles[5].log_prices) == 2
