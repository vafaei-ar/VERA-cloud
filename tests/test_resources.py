from api.services.resources import lookup_resources


def test_county_aliases_and_state_boundary():
    for name in ("McKean", "McKean County", "McKean County, PA"):
        result = lookup_resources(name, "transportation")
        assert not result["used_fallback"]
        assert result["resources"]["transportation"][0]["phone"] == "866-282-4968"
    assert lookup_resources("Monroe County, NY")["used_fallback"]
    assert lookup_resources("Monroe County, PA", "meals")["resources"]["meals"][0]["phone"] == "570-424-8794"


def test_unknown_local_category_is_explicit_not_a_fake_provider():
    result = lookup_resources("monroe", "rehab")
    assert "rehab" in result["coverage_gaps"] and result["note"]
    assert "No verified local" in result["resources"]["rehab"][0]["description"]
    assert result["source_checked_at"] == "2026-09-07"


def test_unrecognized_need_never_returns_unrelated_entries():
    result = lookup_resources("mckean", "unrecognized")
    assert not result["resources"] and "Unknown need" in result["note"]
