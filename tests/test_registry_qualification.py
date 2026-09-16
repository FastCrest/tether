from tether.registry.models import list_families
from tether.registry.qualification import (
    FAMILY_QUALIFICATIONS,
    qualification_for,
    qualification_gaps,
)


def test_every_registry_family_has_an_explicit_qualification():
    assert set(list_families()) == set(FAMILY_QUALIFICATIONS)


def test_qualification_is_conservative_and_copied():
    smolvla = qualification_for("smolvla")
    assert smolvla["training"] == "qualified-lora"
    assert qualification_for("dreamzero")["checkpoint"] == "registry-only"
    smolvla["notes"].append("changed")
    assert "changed" not in qualification_for("smolvla")["notes"]


def test_acceptance_requirements_are_recorded_for_every_family():
    assert all(value.get("acceptance") for value in FAMILY_QUALIFICATIONS.values())
    assert qualification_gaps("pi0")
    assert qualification_gaps("pi05")
