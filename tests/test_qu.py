from preloved.search.qu import QueryUnderstanding

qu = QueryUnderstanding()


def slots(q):
    return qu.parse(q).slots()


def test_reads_the_seller_language_into_canonical_slots():
    assert slots("sac chanel noir matelassé") == {
        "brand": "Chanel",
        "category": "bag",
        "colour": "black",
        "style": "quilted",
    }
    assert slots("borsa gucci vintage") == {"brand": "Gucci", "category": "bag", "style": "vintage"}
    assert slots("zapatos ysl")["brand"] == "Saint Laurent"


def test_a_model_implies_its_brand_and_category():
    assert slots("so kate 39") == {
        "brand": "Christian Louboutin",
        "model": "So Kate",
        "category": "shoes",
        "size": "39",
    }
    assert slots("boy bag")["model"] == "Boy"


def test_typos_are_corrected_only_within_the_edit_budget():
    p = qu.parse("birkn togo")
    assert p.slots()["model"] == "Birkin" and ("birkn", "birkin") in p.fuzzy
    assert "brand" not in slots("xyzzy")


def test_an_ambiguous_word_alone_is_not_a_model():
    assert "model" not in slots("kelly")


def test_nicknames_are_left_to_the_learned_retriever():
    p = qu.parse("bv bag")
    assert "brand" not in p.slots() and "bv" in p.rest
