"""Pro-tier consent flow — first-time TTY prompt + signed receipt.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #1:
data collection is EXPLICIT opt-in. Customer must affirmatively accept
before `--collect-data` records anything to disk.

State machine:
- Receipt missing → prompt the operator on TTY; refuse to proceed in
  non-interactive contexts (CI, daemon mode); customer must run
  `tether serve --pro --collect-data` interactively first.
- Receipt present + valid → silent pass (no prompt repeats).
- Receipt present + invalid (corrupted / wrong customer) → fail loud,
  refuse to start. Operator runs `tether pro consent --reset` to clear.
- GDPR/CCPA: `revoke()` wipes the receipt + the data directory; next
  start re-prompts.

The receipt at `~/.tether/pro_consent.json` carries:
- consent_version (int, currently 1)
- customer_id (str — extracted from the Pro license)
- accepted_at (ISO 8601 UTC)
- accepted_terms_version (str — links to the legal terms doc the
  customer was shown)
- pii_options (dict — face-blur mode, instruction-hash mode, state-raw
  mode at the time of acceptance)

Composition with the Pro license: consent is per-customer (license
provides customer_id); a customer who swaps machines doesn't need to
re-consent unless the receipt is missing on the new machine.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# Bumped on a breaking change to consent semantics (e.g., adding a
# fundamentally different PII category). Old receipts get re-prompted
# when the version drifts. Phase 1 = v1.
CONSENT_VERSION = 1

# Default path for the receipt — customer-disk-only, never sync'd to
# Tether servers. Customers may override via `--pro-consent-path`
# (Phase 1.5 wiring).
DEFAULT_CONSENT_PATH = "~/.tether/pro_consent.json"

# Currently-shipped legal terms document version. Update when the
# customer-facing pitch / data-handling policy text changes.
TERMS_VERSION = "2026-04-25"


@dataclass(frozen=True)
class PIIOptions:
    """Snapshot of the PII-handling choices the customer made at consent
    time. If the customer later changes these via CLI, the receipt's
    snapshot is what was approved — diverging values trigger a re-prompt
    so the operator must reaffirm."""

    face_blur_mode: str  # "blur" | "raw" | "skip"
    instruction_mode: str  # "hashed" | "raw"
    state_mode: str  # "raw" | "hashed"

    def __post_init__(self) -> None:
        if self.face_blur_mode not in ("blur", "raw", "skip"):
            raise ValueError(
                f"face_blur_mode must be blur|raw|skip, got {self.face_blur_mode!r}"
            )
        if self.instruction_mode not in ("hashed", "raw"):
            raise ValueError(
                f"instruction_mode must be hashed|raw, got {self.instruction_mode!r}"
            )
        if self.state_mode not in ("raw", "hashed"):
            raise ValueError(
                f"state_mode must be raw|hashed, got {self.state_mode!r}"
            )


@dataclass(frozen=True)
class ConsentReceipt:
    """Persisted record of customer consent. Frozen — saved once, never
    mutated. Re-prompt = wipe + create new receipt."""

    consent_version: int
    customer_id: str
    accepted_at: str  # ISO 8601 UTC
    accepted_terms_version: str
    pii_options: PIIOptions

    SCHEMA_VERSION: ClassVar[int] = CONSENT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "consent_version": self.consent_version,
            "customer_id": self.customer_id,
            "accepted_at": self.accepted_at,
            "accepted_terms_version": self.accepted_terms_version,
            "pii_options": asdict(self.pii_options),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConsentReceipt":
        return cls(
            consent_version=int(d["consent_version"]),
            customer_id=str(d["customer_id"]),
            accepted_at=str(d["accepted_at"]),
            accepted_terms_version=str(d["accepted_terms_version"]),
            pii_options=PIIOptions(**d["pii_options"]),
        )


class ConsentRequired(Exception):
    """Raised when consent is missing AND we're in a non-interactive
    context. Caller should print a clear message + exit 1; never silently
    proceed without consent."""


class ConsentMismatch(Exception):
    """Raised when the loaded receipt is for a different customer than
    the current Pro license, or when PII options have changed without
    re-prompt. Operator must `tether pro consent --reset`."""


class ProConsent:
    """Consent state machine. One instance per process; load OR create on
    startup; check on /act records.

    Usage:

        consent = ProConsent.load_or_prompt(
            customer_id="acme-corp",
            pii_options=PIIOptions(face_blur_mode="blur",
                                   instruction_mode="hashed",
                                   state_mode="raw"),
            interactive=sys.stdin.isatty(),
        )
        # If consent.has_consent → safe to record
    """

    __slots__ = ("_receipt", "_path")

    def __init__(self, receipt: ConsentReceipt | None, path: Path):
        self._receipt = receipt
        self._path = path

    @property
    def has_consent(self) -> bool:
        return self._receipt is not None

    @property
    def receipt(self) -> ConsentReceipt | None:
        return self._receipt

    @property
    def path(self) -> Path:
        return self._path

    @classmethod
    def load_or_prompt(
        cls,
        *,
        customer_id: str,
        pii_options: PIIOptions,
        path: str | Path = DEFAULT_CONSENT_PATH,
        interactive: bool | None = None,
        prompt_fn=None,
    ) -> "ProConsent":
        """Load existing receipt OR prompt the operator on TTY.

        Args:
            customer_id: from the Pro license. Receipt's customer_id must
                match — refuses to load a receipt that was for a different
                customer.
            pii_options: current CLI-passed PII options. Must match the
                receipt's snapshot. Mismatch → re-prompt.
            path: receipt location.
            interactive: True/False to force; None auto-detects via stdin.isatty().
            prompt_fn: testable injection point for the prompt. Default
                uses input() against stdin/stdout. Receives the prompt
                text + must return True (accepted) / False (rejected).

        Raises:
            ConsentRequired: when no receipt exists AND interactive is False.
            ConsentMismatch: when the receipt is for a different customer.
        """
        path_obj = Path(path).expanduser()
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if path_obj.exists():
            try:
                data = json.loads(path_obj.read_text())
                receipt = ConsentReceipt.from_dict(data)
            except Exception as exc:
                raise ConsentMismatch(
                    f"corrupted consent receipt at {path_obj}: {exc}. "
                    f"Run `tether pro consent --reset` to clear."
                ) from exc
            # Validate against current state.
            if receipt.consent_version != CONSENT_VERSION:
                raise ConsentMismatch(
                    f"consent_version drift: receipt={receipt.consent_version}, "
                    f"current={CONSENT_VERSION}. Re-prompt required."
                )
            if receipt.customer_id != customer_id:
                raise ConsentMismatch(
                    f"customer_id mismatch: receipt={receipt.customer_id!r}, "
                    f"current={customer_id!r}. Receipt is for a different Pro "
                    f"license; reset before continuing."
                )
            if receipt.pii_options != pii_options:
                raise ConsentMismatch(
                    f"PII options changed since acceptance: "
                    f"receipt={receipt.pii_options}, current={pii_options}. "
                    f"Re-prompt required so customer can reaffirm."
                )
            logger.info(
                "Pro consent valid — customer_id=%s accepted_at=%s",
                customer_id, receipt.accepted_at,
            )
            return cls(receipt=receipt, path=path_obj)

        # No receipt → prompt on TTY OR refuse.
        if interactive is None:
            interactive = sys.stdin.isatty() and sys.stdout.isatty()
        if not interactive:
            raise ConsentRequired(
                f"Pro data-collection consent missing at {path_obj}. "
                f"Run `tether serve --pro --collect-data` interactively "
                f"once to accept the data-handling terms; subsequent "
                f"non-interactive starts will succeed silently."
            )

        accepted = (prompt_fn or _default_prompt_fn)(
            _consent_prompt_text(customer_id=customer_id, pii_options=pii_options)
        )
        if not accepted:
            raise ConsentRequired(
                "Customer declined Pro data-collection consent. "
                "--collect-data disabled."
            )

        receipt = ConsentReceipt(
            consent_version=CONSENT_VERSION,
            customer_id=customer_id,
            accepted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            accepted_terms_version=TERMS_VERSION,
            pii_options=pii_options,
        )
        # Atomic write
        tmp = path_obj.with_suffix(path_obj.suffix + ".tmp")
        tmp.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path_obj)
        os.chmod(path_obj, 0o600)  # customer-private
        logger.info(
            "Pro consent recorded — customer_id=%s saved to %s",
            customer_id, path_obj,
        )
        return cls(receipt=receipt, path=path_obj)

    def revoke(self, *, also_wipe_data_dir: Path | None = None) -> None:
        """GDPR/CCPA delete: wipe the receipt + optionally the collected
        data dir. Next start re-prompts. Idempotent — second call is no-op."""
        if self._path.exists():
            self._path.unlink()
            logger.warning("Pro consent revoked — receipt removed at %s", self._path)
        self._receipt = None
        if also_wipe_data_dir is not None:
            data_dir = Path(also_wipe_data_dir).expanduser()
            if data_dir.exists():
                import shutil
                shutil.rmtree(data_dir, ignore_errors=True)
                logger.warning(
                    "Pro consent revoked — data directory removed at %s",
                    data_dir,
                )


def _default_prompt_fn(text: str) -> bool:
    """Default TTY prompt — prints text + reads a single Y/n."""
    print(text)
    response = input("Accept? [y/N]: ").strip().lower()
    return response in ("y", "yes")


def _consent_prompt_text(*, customer_id: str, pii_options: PIIOptions) -> str:
    """The text the customer sees on first --collect-data start. Edit when
    the legal terms / data-handling policy changes; bump TERMS_VERSION."""
    return f"""
═══════════════════════════════════════════════════════════════════════
Tether Pro — Data Collection Consent
═══════════════════════════════════════════════════════════════════════

Customer ID: {customer_id}
Terms version: {TERMS_VERSION}
Tether version: (see `tether --version`)

You're about to enable customer-data collection for the self-distilling
training loop. Per the Tether data-handling policy:

  - All data lives on YOUR disk (default ~/.tether/pro-data/). Tether
    never ingests it. The distill pipeline reads it locally OR uploads
    to YOUR private HF Hub repo (not Tether's).
  - 90-day rolling retention; older data auto-pruned.
  - PII handling for this run:
      face_blur_mode    = {pii_options.face_blur_mode}
      instruction_mode  = {pii_options.instruction_mode}
      state_mode        = {pii_options.state_mode}
  - You can revoke at any time via `tether pro consent --revoke`,
    which wipes the receipt + the collected data directory.

By accepting, you confirm:
  1. You're authorized to collect this data.
  2. The PII options above match your privacy/compliance needs.
  3. You'll re-prompt yourself if you change the PII options later.

═══════════════════════════════════════════════════════════════════════
"""


__all__ = [
    "CONSENT_VERSION",
    "DEFAULT_CONSENT_PATH",
    "TERMS_VERSION",
    "ConsentMismatch",
    "ConsentReceipt",
    "ConsentRequired",
    "PIIOptions",
    "ProConsent",
]
