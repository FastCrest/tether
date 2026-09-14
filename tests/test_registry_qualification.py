from tether.registry.models import list_families
from tether.registry.qualification import FAMILY_QUALIFICATIONS, qualification_for


def test_every_registry_family_has_an_explicit_qualification():
    assert set(list_families()) == set(FAMILY_QUALIFICATIONS)


def test_qualification_is_conservative_and_copied():
    smolvla = qualification_for("smolvla")
    assert smolvla["training"] == "qualified-lora"
    assert qualification_for("dreamzero")["checkpoint"] == "registry-only"
    smolvla["notes"].append("changed")
    assert "changed" not in qualification_for("smolvla")["notes"]
