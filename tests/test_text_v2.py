from sweet_truth.text import Resources, classify_ingredients


def test_v2_flags_sweeteners_it_was_not_trained_on(resources_v2: Resources):
    text = "Wheat flour, Isomaltulose, Coconut sugar (12%), Salt and Water."
    result = classify_ingredients(text, resources_v2)

    assert [item.text for item in result.items] == [
        "Wheat flour",
        "Isomaltulose",
        "Coconut sugar (12%)",
        "Salt",
        "Water",
    ]
    assert [item.text for item in result.sweeteners] == ["Isomaltulose", "Coconut sugar (12%)"]
    for item in result.items:
        assert text[item.start : item.end] == item.text
        assert 0.5 <= item.confidence <= 1


def test_v2_explains_each_sweetener(resources_v2: Resources):
    result = classify_ingredients("Sorbitol, Salt", resources_v2)
    sorbitol, salt = result.items
    assert sorbitol.reasons and any("tol" in reason for reason in sorbitol.reasons)
    assert salt.reasons == []


def test_v2_keeps_bracketed_codes_with_their_ingredient(resources_v2: Resources):
    result = classify_ingredients("Sweeteners (951, 950), Cocoa solids", resources_v2)
    assert [item.text for item in result.items] == ["Sweeteners (951, 950)", "Cocoa solids"]
    assert [item.text for item in result.sweeteners] == ["Sweeteners (951, 950)"]


def test_v2_does_not_need_spacy(resources_v2: Resources):
    assert resources_v2.nlp is None
    assert resources_v2.kind == "v2"


def test_v2_splits_where_a_photo_lost_the_comma_after_a_bracket(resources_v2: Resources):
    text = 'Liquid glucose [1%"] Permitted emulsifiers [322, 471], Salt'
    result = classify_ingredients(text, resources_v2)
    assert [item.text for item in result.items] == [
        'Liquid glucose [1%"]',
        "Permitted emulsifiers [322, 471]",
        "Salt",
    ]
    assert [item.text for item in result.sweeteners] == ['Liquid glucose [1%"]']
