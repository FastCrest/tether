"""Emit the installed runtime inventory for advisory lookup without re-resolution.

PyTorch CPU wheels have a local +cpu suffix. Advisory services track the public
upstream version, so retain that exact version while removing only the suffix.
The checked-out project itself is not a published third-party dependency.
"""

from __future__ import annotations

from importlib.metadata import distributions


def inventory(distributions_to_scan) -> list[str]:
    requirements = []
    for distribution in distributions_to_scan:
        name = distribution.metadata["Name"]
        if name.lower().replace("_", "-") == "fastcrest-tether":
            continue
        requirements.append(f"{name}=={distribution.version.split('+', 1)[0]}")
    return sorted(requirements, key=str.lower)


if __name__ == "__main__":
    print("\n".join(inventory(distributions())))
