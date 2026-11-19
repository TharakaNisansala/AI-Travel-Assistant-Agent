"""party.py: ticket categories for who is travelling, decided by code."""

from __future__ import annotations

import pytest

from ai_travel_assistant.party import (
    complete_party,
    describe_group,
    group_breakdown,
    missing_group_fields,
    nationality_group,
)


@pytest.mark.parametrize(
    ("nationality", "group"),
    [
        ("Sri Lankan", "local"),
        ("sri lanka citizen", "local"),
        ("Indian", "saarc"),
        ("Maldivian", "saarc"),
        ("Nepalese", "saarc"),
        ("Pakistan", "saarc"),
        ("German", "non_saarc"),
        ("British", "non_saarc"),
        ("Chinese", "non_saarc"),  # Asian, but not a SAARC member
        ("SAARC", "saarc"),
        ("non-SAARC", "non_saarc"),
        # Not enough to pick a price: the agent must ask.
        ("dual", "unknown"),
        ("Sri Lankan-British dual citizen", "unknown"),
        ("foreigner", "unknown"),
        ("tourist", "unknown"),
        ("Asian", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_nationality_group(nationality, group):
    assert nationality_group(nationality) == group


def member(nationality, count=1, age="adult"):
    return {"nationality": nationality, "age_group": age, "count": count}


def test_a_mixed_party_is_counted_per_category_and_age():
    trip = {
        "party": [
            member("German"),
            member("Sri Lankan", 2),
            member("Indian"),
            member("Sri Lankan", 1, "child"),
        ]
    }
    assert group_breakdown(trip) == [
        ("local", "adult", 2),
        ("local", "child", 1),
        ("saarc", "adult", 1),
        ("non_saarc", "adult", 1),
    ]
    assert describe_group(trip) == (
        "2 Sri Lankan adults, 1 Sri Lankan child, 1 SAARC adult, "
        "1 non-SAARC foreign adult"
    )


def test_a_single_citizenship_is_split_into_adults_and_children():
    trip = {"citizenship": "non_saarc", "travellers": 4, "children": 1}
    assert group_breakdown(trip) == [
        ("non_saarc", "adult", 3),
        ("non_saarc", "child", 1),
    ]


def test_nothing_is_counted_until_the_group_is_known():
    assert group_breakdown({"travellers": 3}) == []
    assert group_breakdown({"citizenship": "mixed", "travellers": 3}) == []
    assert group_breakdown(None) == []


def test_the_party_list_sets_travellers_and_children():
    trip = complete_party(
        {
            "travellers": 9,
            "party": [member("Sri Lankan", 2), member("Indian", 1, "child")],
        }
    )
    assert (trip["travellers"], trip["children"]) == (3, 1)


def test_one_traveller_is_an_adult():
    assert complete_party({"travellers": 1})["children"] == 0
    assert complete_party({"travellers": 2}).get("children") is None


@pytest.mark.parametrize(
    ("trip", "missing"),
    [
        ({"travellers": 2}, ["citizenship"]),
        ({"travellers": 2, "citizenship": "mixed"}, ["party"]),
        ({"travellers": 2, "citizenship": "local"}, ["children"]),
        ({"travellers": 2, "citizenship": "local", "children": 0}, []),
        ({"travellers": 1, "citizenship": "saarc", "children": 0}, []),
        ({"party": [member("Sri Lankan", 2)]}, []),
        ({"party": [member("Sri Lankan"), member("dual")]}, ["party"]),
        ({"party": [member("tourist", 2)]}, ["party"]),
    ],
)
def test_missing_group_fields(trip, missing):
    assert missing_group_fields(trip) == missing
