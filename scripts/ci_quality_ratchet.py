#!/usr/bin/env python3
"""Fail CI only when a commit adds Ruff or mypy debt.

The protected base commit is the baseline. No checked-in finding inventory is
read, so a pull request cannot authorize its own debt by editing a snapshot.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Sequence

MAX_RELOCATION_LINES = 50
TRUSTED_POLICY_REVIEWERS = frozenset({"rylinjames"})
AUTHORIZATION_PATHS = frozenset(
    {
        ".github/CODEOWNERS",
        ".github/workflows/quality-baseline-update.yml",
        ".github/workflows/quality-ratchet.yml",
        "scripts/ci_quality_ratchet.py",
    }
)
POLICY_WORKFLOW_PATH = ".github/workflows/quality-baseline-update.yml"
_APPROVAL_EVIDENCE_FIELDS = frozenset(
    {
        "base_sha",
        "candidate_sha",
        "repository",
        "run_id",
        "schema_version",
        "validation_artifact_digest",
        "validation_artifact_id",
        "workflow_path",
    }
)
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_MYPY_LINE = re.compile(r"^(.*?):(\d+):(\d+):\s+(error|warning|note):\s+(.*?)(?:\s+\[([^\]]+)\])?$")
_RUFF_FORMAT_LINE = re.compile(r"^Would reformat:\s+(.+?)\s*$")
_DIFF_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class RatchetError(RuntimeError):
    """Raised when a tool or repository operation cannot be trusted."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One normalized static-analysis finding."""

    tool: str
    rule: str
    path: str
    message: str
    line: int
    column: int

    @property
    def fingerprint(self) -> tuple[str, str, str, str]:
        """Return the stable identity; source coordinates are metadata only."""
        return (self.tool, self.rule, self.path, self.message)


@dataclass(frozen=True, slots=True)
class DiffHunk:
    """A zero-context Git diff hunk used to relocate a base line."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int


@dataclass(frozen=True, slots=True)
class Comparison:
    """The monotonicity decision and its evidence."""

    new_findings: tuple[Finding, ...]
    removed_findings: tuple[Finding, ...]
    relocated_matches: int
    base_count: int
    head_count: int
    base_counts_by_tool: dict[str, int]
    head_counts_by_tool: dict[str, int]

    @property
    def passed(self) -> bool:
        """Return whether both the stable multiset and counts are monotonic."""
        return (
            not self.new_findings
            and self.head_count <= self.base_count
            and all(
                self.head_counts_by_tool.get(tool, 0) <= count
                for tool, count in self.base_counts_by_tool.items()
            )
        )


def normalize_path(raw_path: str, *, checkout: Path) -> str:
    """Normalize a tool path to a repository-relative POSIX path."""
    text = raw_path.strip().replace("\\", "/")
    root = checkout.resolve().as_posix().rstrip("/")
    if text == root:
        return "."
    if text.startswith(root + "/"):
        text = text[len(root) + 1 :]
    while text.startswith("./"):
        text = text[2:]
    normalized = PurePosixPath(text).as_posix()
    return normalized or "."


def normalize_message(message: str) -> str:
    """Collapse presentation-only whitespace and volatile line references."""
    value = " ".join(message.strip().split())
    return re.sub(r"\bline\s+\d+\b", "line <line>", value, flags=re.IGNORECASE)


def verify_policy_approval_evidence(
    value: object,
    *,
    expected_candidate_sha: str,
    expected_base_sha: str,
    expected_run_id: int,
    expected_repository: str,
) -> None:
    """Verify the immutable manual-run envelope for one exact comparison."""
    if not isinstance(value, dict) or set(value) != _APPROVAL_EVIDENCE_FIELDS:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RatchetError(f"policy approval evidence fields mismatch: {actual}")
    if value["schema_version"] != 1:
        raise RatchetError("unsupported policy approval evidence schema")
    if value["candidate_sha"] != expected_candidate_sha:
        raise RatchetError("policy approval candidate SHA does not match head")
    if value["base_sha"] != expected_base_sha:
        raise RatchetError("policy approval base SHA does not match protected base")
    if value["run_id"] != expected_run_id or isinstance(value["run_id"], bool):
        raise RatchetError("policy approval run id does not match status target")
    if value["repository"] != expected_repository:
        raise RatchetError("policy approval repository does not match")
    if value["workflow_path"] != POLICY_WORKFLOW_PATH:
        raise RatchetError("policy approval workflow path does not match")
    artifact_id = value["validation_artifact_id"]
    if isinstance(artifact_id, bool) or not isinstance(artifact_id, int) or artifact_id <= 0:
        raise RatchetError("policy approval validation artifact id is invalid")
    digest = value["validation_artifact_digest"]
    if not isinstance(digest, str) or not _SHA256_DIGEST.fullmatch(digest):
        raise RatchetError("policy approval validation artifact digest is invalid")


def verify_current_main_binding(
    ref_value: object,
    *,
    expected_main_sha: str,
    pull_request_value: object | None = None,
    expected_head_sha: str | None = None,
) -> None:
    """Reject stale workflow events by binding them to the live main ref."""
    if not _COMMIT_SHA.fullmatch(expected_main_sha):
        raise RatchetError("expected protected-main SHA is invalid")
    if not isinstance(ref_value, dict) or ref_value.get("ref") != "refs/heads/main":
        raise RatchetError("protected-main ref response is invalid")
    ref_object = ref_value.get("object")
    if not isinstance(ref_object, dict) or ref_object.get("type") != "commit":
        raise RatchetError("protected-main ref does not resolve to a commit")
    if ref_object.get("sha") != expected_main_sha:
        raise RatchetError("workflow event is stale because protected main advanced")

    if pull_request_value is None:
        if expected_head_sha is not None:
            raise RatchetError("pull-request head was provided without pull-request data")
        return
    if expected_head_sha is None or not _COMMIT_SHA.fullmatch(expected_head_sha):
        raise RatchetError("expected pull-request head SHA is invalid")
    if not isinstance(pull_request_value, dict) or pull_request_value.get("state") != "open":
        raise RatchetError("pull request is no longer open")
    base = pull_request_value.get("base")
    head = pull_request_value.get("head")
    if not isinstance(base, dict) or base.get("ref") != "main":
        raise RatchetError("pull request no longer targets protected main")
    if base.get("sha") != expected_main_sha:
        raise RatchetError("pull-request base SHA does not match current protected main")
    if not isinstance(head, dict) or head.get("sha") != expected_head_sha:
        raise RatchetError("pull-request head SHA no longer matches the workflow event")


def verify_policy_environment(value: object) -> None:
    """Require the exact protected environment reviewer allowlist."""
    if not isinstance(value, dict):
        raise RatchetError("policy environment response is not an object")
    rules = value.get("protection_rules")
    if not isinstance(rules, list):
        raise RatchetError("policy environment protection rules are missing")
    reviewer_rules = [
        rule
        for rule in rules
        if isinstance(rule, dict) and rule.get("type") == "required_reviewers"
    ]
    if len(reviewer_rules) != 1:
        raise RatchetError("policy environment must have exactly one reviewer rule")
    reviewers = reviewer_rules[0].get("reviewers")
    if not isinstance(reviewers, list):
        raise RatchetError("policy environment reviewer list is missing")
    actual_reviewers: list[str] = []
    for entry in reviewers:
        if not isinstance(entry, dict) or entry.get("type") != "User":
            raise RatchetError("policy environment reviewers must be individual users")
        reviewer = entry.get("reviewer")
        login = reviewer.get("login") if isinstance(reviewer, dict) else None
        if not isinstance(login, str) or not login:
            raise RatchetError("policy environment reviewer login is invalid")
        actual_reviewers.append(login.casefold())
    if len(actual_reviewers) != len(set(actual_reviewers)):
        raise RatchetError("policy environment reviewer list contains duplicates")
    if frozenset(actual_reviewers) != TRUSTED_POLICY_REVIEWERS:
        raise RatchetError("policy environment reviewer allowlist does not match")

    branch_policy = value.get("deployment_branch_policy")
    if not isinstance(branch_policy, dict):
        raise RatchetError("policy environment deployment branch policy is missing")
    if (
        branch_policy.get("protected_branches") is not True
        or branch_policy.get("custom_branch_policies") is not False
    ):
        raise RatchetError("policy environment must allow protected branches only")


def parse_ruff_check(output: str, *, checkout: Path) -> list[Finding]:
    """Parse ``ruff check --output-format=json`` output."""
    try:
        values = json.loads(output or "[]")
    except json.JSONDecodeError as exc:
        raise RatchetError(f"Ruff check emitted invalid JSON: {exc}") from exc
    if not isinstance(values, list):
        raise RatchetError("Ruff check JSON must be a list")

    findings: list[Finding] = []
    for value in values:
        if not isinstance(value, dict):
            raise RatchetError("Ruff check JSON contained a non-object finding")
        location = value.get("location") or {}
        if not isinstance(location, dict):
            location = {}
        findings.append(
            Finding(
                tool="ruff-check",
                rule=str(value.get("code") or "syntax"),
                path=normalize_path(str(value.get("filename") or "<unknown>"), checkout=checkout),
                message=normalize_message(str(value.get("message") or "unknown Ruff finding")),
                line=int(location.get("row") or 1),
                column=int(location.get("column") or 1),
            )
        )
    return findings


def parse_ruff_format(output: str, *, checkout: Path) -> list[Finding]:
    """Parse ``ruff format --check`` output into one finding per file."""
    findings: list[Finding] = []
    for line in output.splitlines():
        match = _RUFF_FORMAT_LINE.match(line.strip())
        if match:
            findings.append(
                Finding(
                    tool="ruff-format",
                    rule="format",
                    path=normalize_path(match.group(1), checkout=checkout),
                    message="File is not formatted",
                    line=1,
                    column=1,
                )
            )
    return findings


def parse_mypy(output: str, *, checkout: Path) -> list[Finding]:
    """Parse stable, non-pretty mypy error lines and ignore explanatory notes."""
    findings: list[Finding] = []
    for line in output.splitlines():
        match = _MYPY_LINE.match(line.strip())
        if not match or match.group(4) != "error":
            continue
        findings.append(
            Finding(
                tool="mypy",
                rule=match.group(6) or "error",
                path=normalize_path(match.group(1), checkout=checkout),
                message=normalize_message(match.group(5)),
                line=int(match.group(2)),
                column=int(match.group(3)),
            )
        )
    return findings


def parse_diff_hunks(diff_text: str) -> list[DiffHunk]:
    """Parse zero-context unified diff hunk headers."""
    hunks: list[DiffHunk] = []
    for line in diff_text.splitlines():
        match = _DIFF_HUNK.match(line)
        if match:
            hunks.append(
                DiffHunk(
                    old_start=int(match.group(1)),
                    old_count=int(match.group(2) or 1),
                    new_start=int(match.group(3)),
                    new_count=int(match.group(4) or 1),
                )
            )
    return hunks


def relocate_base_line(line: int, hunks: Sequence[DiffHunk]) -> int:
    """Map a base coordinate through a diff, including large pure insertions."""
    delta = 0
    for hunk in hunks:
        if hunk.old_count == 0:
            if line > hunk.old_start:
                delta += hunk.new_count
                continue
            break
        old_end = hunk.old_start + hunk.old_count - 1
        if line < hunk.old_start:
            break
        if line <= old_end:
            offset = line - hunk.old_start
            if hunk.new_count == 0:
                return hunk.new_start
            return hunk.new_start + min(offset, hunk.new_count - 1)
        delta += hunk.new_count - hunk.old_count
    return max(1, line + delta)


def _pair_group(
    base: Sequence[Finding],
    head: Sequence[Finding],
    hunks: Sequence[DiffHunk],
) -> tuple[list[Finding], list[Finding], int]:
    """Pair one fingerprint multiset, preferring bounded diff relocation."""
    unmatched_base = list(sorted(base, key=lambda item: (item.line, item.column)))
    unmatched_head = list(sorted(head, key=lambda item: (item.line, item.column)))
    relocated = 0

    for base_finding in list(unmatched_base):
        expected = relocate_base_line(base_finding.line, hunks)
        candidates = [
            item for item in unmatched_head if abs(item.line - expected) <= MAX_RELOCATION_LINES
        ]
        if not candidates:
            continue
        selected = min(candidates, key=lambda item: (abs(item.line - expected), item.column))
        unmatched_base.remove(base_finding)
        unmatched_head.remove(selected)
        if selected.line != base_finding.line:
            relocated += 1

    # Coordinates are not part of identity. Pair any remaining identical
    # fingerprints deterministically; only surplus occurrences are new debt.
    fallback_pairs = min(len(unmatched_base), len(unmatched_head))
    if fallback_pairs:
        del unmatched_base[:fallback_pairs]
        del unmatched_head[:fallback_pairs]
    return unmatched_head, unmatched_base, relocated


def compare_findings(
    base: Sequence[Finding],
    head: Sequence[Finding],
    *,
    diff_hunks: Callable[[str], Sequence[DiffHunk]],
) -> Comparison:
    """Compare stable finding multisets and enforce monotonic non-increase."""
    base_groups: dict[tuple[str, str, str, str], list[Finding]] = defaultdict(list)
    head_groups: dict[tuple[str, str, str, str], list[Finding]] = defaultdict(list)
    for finding in base:
        base_groups[finding.fingerprint].append(finding)
    for finding in head:
        head_groups[finding.fingerprint].append(finding)

    new: list[Finding] = []
    removed: list[Finding] = []
    relocated = 0
    for fingerprint in sorted(set(base_groups) | set(head_groups)):
        base_group = base_groups.get(fingerprint, [])
        head_group = head_groups.get(fingerprint, [])
        hunks = diff_hunks(fingerprint[2]) if base_group and head_group else ()
        group_new, group_removed, group_relocated = _pair_group(base_group, head_group, hunks)
        new.extend(group_new)
        removed.extend(group_removed)
        relocated += group_relocated

    return Comparison(
        new_findings=tuple(
            sorted(new, key=lambda item: (item.tool, item.path, item.line, item.rule))
        ),
        removed_findings=tuple(
            sorted(removed, key=lambda item: (item.tool, item.path, item.line, item.rule))
        ),
        relocated_matches=relocated,
        base_count=len(base),
        head_count=len(head),
        base_counts_by_tool=dict(Counter(item.tool for item in base)),
        head_counts_by_tool=dict(Counter(item.tool for item in head)),
    )


def is_baseline_artifact(path: str) -> bool:
    """Return whether a changed path is reserved for mutable debt snapshots."""
    normalized = PurePosixPath(path.lower()).as_posix()
    name = PurePosixPath(normalized).name
    return (
        normalized.startswith(".ci/quality-baseline/")
        or normalized.startswith(".github/quality-baseline/")
        or name.startswith(".quality-baseline")
        or bool(re.fullmatch(r"quality[-_.]baselines?\.(json|toml|ya?ml|txt)", name))
    )


def _changed_paths(repo: Path, base_sha: str, head_sha: str) -> list[str]:
    """List paths changed between two explicit commits."""
    result = _run(
        ["git", "diff", "--name-only", "-z", base_sha, head_sha],
        cwd=repo,
        accepted={0},
    )
    return sorted(path for path in result.stdout.split("\0") if path)


def _git_text(repo: Path, sha: str, path: str) -> str:
    """Read one UTF-8 text file exactly from a commit."""
    return _run(["git", "show", f"{sha}:{path}"], cwd=repo, accepted={0}).stdout


def validate_tool_policy_text(pyproject_text: str) -> dict[str, dict[str, object]]:
    """Parse Ruff/mypy policy as data without importing configured plugins."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise RatchetError("Python 3.11+ or tomli is required to parse tool policy") from exc
    try:
        value = tomllib.loads(pyproject_text)
    except Exception as exc:
        raise RatchetError(f"pyproject.toml is not valid TOML: {exc}") from exc
    tool = value.get("tool", {})
    if not isinstance(tool, dict):
        raise RatchetError("pyproject.toml [tool] must be a table")
    policies: dict[str, dict[str, object]] = {}
    for name in ("ruff", "mypy"):
        policy = tool.get(name, {})
        if not isinstance(policy, dict):
            raise RatchetError(f"pyproject.toml [tool.{name}] must be a table")
        policies[name] = policy
    if "plugins" in policies["mypy"]:
        raise RatchetError("mypy plugins are forbidden in the protected quality policy")
    return policies


def changed_authorization_inputs(repo: Path, base_sha: str, head_sha: str) -> list[str]:
    """List gate, ownership, mutable baseline, and effective-policy changes."""
    changed = _changed_paths(repo, base_sha, head_sha)
    authorization_changes = [path for path in changed if path in AUTHORIZATION_PATHS]
    authorization_changes.extend(path for path in changed if is_baseline_artifact(path))

    if "pyproject.toml" in changed:
        base_policy = validate_tool_policy_text(_git_text(repo, base_sha, "pyproject.toml"))
        head_policy = validate_tool_policy_text(_git_text(repo, head_sha, "pyproject.toml"))
        for tool in ("ruff", "mypy"):
            if base_policy[tool] != head_policy[tool]:
                authorization_changes.append(f"pyproject.toml#[tool.{tool}]")
    return sorted(set(authorization_changes))


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    accepted: set[int],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in accepted:
        rendered = " ".join(command)
        detail = (result.stdout + "\n" + result.stderr).strip()[-4000:]
        raise RatchetError(f"command failed ({result.returncode}): {rendered}\n{detail}")
    return result


def _tool_version(module: str, *, cwd: Path) -> str:
    result = _run(
        [sys.executable, "-I", "-m", module, "--version"],
        cwd=cwd,
        accepted={0},
    )
    return (result.stdout or result.stderr).strip()


def collect_findings(
    checkout: Path,
    *,
    cache_dir: Path,
    protected_pyproject: str,
) -> tuple[list[Finding], dict[str, str]]:
    """Run all three tools in one checkout and parse nonzero finding exits."""
    # Both snapshots must use the protected base policy. These worktrees are
    # disposable, so replacing a PR-controlled pyproject cannot mutate a commit.
    policy_path = checkout / "pyproject.toml"
    policy_path.unlink()
    policy_path.write_text(protected_pyproject)
    ruff_check = _run(
        [
            sys.executable,
            "-I",
            "-m",
            "ruff",
            "check",
            "--config=pyproject.toml",
            "--no-cache",
            "--output-format=json",
            ".",
        ],
        cwd=checkout,
        accepted={0, 1},
    )
    ruff_check_findings = parse_ruff_check(ruff_check.stdout, checkout=checkout)
    if ruff_check.returncode == 1 and not ruff_check_findings:
        raise RatchetError("Ruff check returned findings but its JSON parsed empty")

    ruff_format = _run(
        [
            sys.executable,
            "-I",
            "-m",
            "ruff",
            "format",
            "--config=pyproject.toml",
            "--check",
            ".",
        ],
        cwd=checkout,
        accepted={0, 1},
    )
    ruff_format_output = ruff_format.stdout + "\n" + ruff_format.stderr
    ruff_format_findings = parse_ruff_format(ruff_format_output, checkout=checkout)
    if ruff_format.returncode == 1 and not ruff_format_findings:
        raise RatchetError("Ruff format returned findings but no file paths were parsed")

    mypy = _run(
        [
            sys.executable,
            "-I",
            "-m",
            "mypy",
            "--config-file=pyproject.toml",
            "--show-error-codes",
            "--show-column-numbers",
            "--no-pretty",
            "--no-color-output",
            "--no-error-summary",
            f"--cache-dir={cache_dir}",
            "src/",
        ],
        cwd=checkout,
        accepted={0, 1},
    )
    mypy_output = mypy.stdout + "\n" + mypy.stderr
    mypy_findings = parse_mypy(mypy_output, checkout=checkout)
    if mypy.returncode == 1 and not mypy_findings:
        raise RatchetError("mypy returned findings but no error lines were parsed")

    return (
        ruff_check_findings + ruff_format_findings + mypy_findings,
        {
            "ruff": _tool_version("ruff", cwd=checkout),
            "mypy": _tool_version("mypy", cwd=checkout),
        },
    )


@contextmanager
def detached_worktree(repo: Path, sha: str, destination: Path) -> Iterator[Path]:
    """Materialize an immutable commit without altering the caller's checkout."""
    _run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo, accepted={0})
    _run(
        ["git", "worktree", "add", "--detach", str(destination), sha],
        cwd=repo,
        accepted={0},
    )
    try:
        yield destination
    finally:
        _run(
            ["git", "worktree", "remove", "--force", str(destination)],
            cwd=repo,
            accepted={0},
        )


def _diff_hunk_provider(
    repo: Path, base_sha: str, head_sha: str
) -> Callable[[str], list[DiffHunk]]:
    cache: dict[str, list[DiffHunk]] = {}
    changed_result = _run(
        ["git", "diff", "--no-renames", "--name-only", "-z", base_sha, head_sha],
        cwd=repo,
        accepted={0},
    )
    changed_paths = {path for path in changed_result.stdout.split("\0") if path}

    def provide(path: str) -> list[DiffHunk]:
        if path not in changed_paths:
            return []
        if path not in cache:
            result = _run(
                [
                    "git",
                    "diff",
                    "--no-ext-diff",
                    "--no-renames",
                    "--unified=0",
                    base_sha,
                    head_sha,
                    "--",
                    path,
                ],
                cwd=repo,
                accepted={0},
            )
            cache[path] = parse_diff_hunks(result.stdout)
        return cache[path]

    return provide


def _finding_dict(finding: Finding) -> dict[str, object]:
    value = asdict(finding)
    value["fingerprint"] = list(finding.fingerprint)
    return value


def run_ratchet(
    *,
    repo: Path,
    base_sha: str,
    head_sha: str,
    report_path: Path,
    authorization_approved: bool = False,
) -> bool:
    """Collect both commits independently, compare them, and write evidence."""
    repo = repo.resolve()
    authorization_changes = changed_authorization_inputs(repo, base_sha, head_sha)
    with tempfile.TemporaryDirectory(prefix="tether-quality-ratchet-") as temporary:
        temporary_root = Path(temporary)
        with detached_worktree(repo, base_sha, temporary_root / "base") as base_checkout:
            protected_pyproject = (base_checkout / "pyproject.toml").read_text()
            validate_tool_policy_text(protected_pyproject)
            base_findings, base_versions = collect_findings(
                base_checkout,
                cache_dir=temporary_root / "mypy-base",
                protected_pyproject=protected_pyproject,
            )
        with detached_worktree(repo, head_sha, temporary_root / "head") as head_checkout:
            head_findings, head_versions = collect_findings(
                head_checkout,
                cache_dir=temporary_root / "mypy-head",
                protected_pyproject=protected_pyproject,
            )

    comparison = compare_findings(
        base_findings,
        head_findings,
        diff_hunks=_diff_hunk_provider(repo, base_sha, head_sha),
    )
    authorization_allowed = not authorization_changes or authorization_approved
    passed = comparison.passed and authorization_allowed
    report = {
        "schema_version": 1,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "passed": passed,
        "authorization_approved": authorization_approved,
        "authorization_input_changes": authorization_changes,
        "base_count": comparison.base_count,
        "head_count": comparison.head_count,
        "base_counts_by_tool": comparison.base_counts_by_tool,
        "head_counts_by_tool": comparison.head_counts_by_tool,
        "new_findings": [_finding_dict(item) for item in comparison.new_findings],
        "removed_findings": [_finding_dict(item) for item in comparison.removed_findings],
        "relocated_matches": comparison.relocated_matches,
        "tool_versions": {"base": base_versions, "head": head_versions},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    status = "PASS" if passed else "FAIL"
    print(
        f"quality ratchet {status}: {comparison.base_count} base -> "
        f"{comparison.head_count} head; {len(comparison.new_findings)} new, "
        f"{len(comparison.removed_findings)} removed"
    )
    for finding in comparison.new_findings:
        print(
            f"NEW {finding.tool} {finding.path}:{finding.line}:{finding.column} "
            f"[{finding.rule}] {finding.message}"
        )
    if authorization_changes and not authorization_approved:
        for path in authorization_changes:
            print(f"PROTECTED authorization input requires manual approval: {path}")
    print(f"report: {report_path}")
    return passed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--authorization-approved",
        action="store_true",
        help="Accept gate/policy changes approved by the protected manual workflow.",
    )
    parser.add_argument("--verify-approval-evidence", type=Path)
    parser.add_argument("--validate-policy-syntax", type=Path)
    parser.add_argument("--verify-main-ref-response", type=Path)
    parser.add_argument("--verify-policy-environment", type=Path)
    parser.add_argument("--verify-pull-request-response", type=Path)
    parser.add_argument("--expected-current-main-sha")
    parser.add_argument("--expected-current-head-sha")
    parser.add_argument("--expected-candidate-sha")
    parser.add_argument("--expected-base-sha")
    parser.add_argument("--expected-run-id", type=int)
    parser.add_argument("--expected-repository")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line quality ratchet."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify_policy_environment is not None:
            try:
                environment_value = json.loads(args.verify_policy_environment.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RatchetError(f"policy environment evidence is unreadable: {exc}") from exc
            verify_policy_environment(environment_value)
            print("policy environment reviewer allowlist verified")
            return 0
        if args.verify_main_ref_response is not None:
            if args.expected_current_main_sha is None:
                parser.error("current-main verification requires --expected-current-main-sha")
            try:
                ref_value = json.loads(args.verify_main_ref_response.read_text())
                pull_request_value = (
                    json.loads(args.verify_pull_request_response.read_text())
                    if args.verify_pull_request_response is not None
                    else None
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RatchetError(f"current-main API evidence is unreadable: {exc}") from exc
            verify_current_main_binding(
                ref_value,
                expected_main_sha=args.expected_current_main_sha,
                pull_request_value=pull_request_value,
                expected_head_sha=args.expected_current_head_sha,
            )
            print("workflow event is bound to the current protected-main tip")
            return 0
        if args.validate_policy_syntax is not None:
            try:
                policy_text = args.validate_policy_syntax.read_text()
            except OSError as exc:
                raise RatchetError(f"candidate tool policy is unreadable: {exc}") from exc
            validate_tool_policy_text(policy_text)
            print("candidate Ruff/mypy policy parsed without executing plugins")
            return 0
        if args.verify_approval_evidence is not None:
            expected = (
                args.expected_candidate_sha,
                args.expected_base_sha,
                args.expected_run_id,
                args.expected_repository,
            )
            if any(value is None for value in expected):
                parser.error("approval evidence verification requires every --expected-* value")
            try:
                evidence = json.loads(args.verify_approval_evidence.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RatchetError(f"policy approval evidence is unreadable: {exc}") from exc
            verify_policy_approval_evidence(
                evidence,
                expected_candidate_sha=args.expected_candidate_sha,
                expected_base_sha=args.expected_base_sha,
                expected_run_id=args.expected_run_id,
                expected_repository=args.expected_repository,
            )
            print("policy approval evidence verified")
            return 0
        if args.base_sha is None or args.head_sha is None or args.report is None:
            parser.error("ratchet comparison requires --base-sha, --head-sha, and --report")
        passed = run_ratchet(
            repo=args.repo,
            base_sha=args.base_sha,
            head_sha=args.head_sha,
            report_path=args.report,
            authorization_approved=args.authorization_approved,
        )
    except RatchetError as exc:
        print(f"quality ratchet operational failure: {exc}", file=sys.stderr)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
