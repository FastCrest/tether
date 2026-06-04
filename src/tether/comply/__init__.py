"""Tether Comply evidence-pack generator.

Comply turns Tether runtime evidence into an offline conformity bundle for
regulated robot makers. It does not declare CE conformity; it produces the
signed evidence file a manufacturer and auditor can inspect.
"""

from tether.comply.export import export_conformity_bundle, verify_conformity_bundle

__all__ = ["export_conformity_bundle", "verify_conformity_bundle"]
