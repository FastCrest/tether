# CI quality ratchet

The quality ratchet compares Ruff check, Ruff format, and mypy findings at the
protected base commit and the exact pull-request head. It recomputes both sides
under the protected base's tool policy; there is no checked-in debt snapshot.

## Repository settings

After the bootstrap merge, repository administrators must:

1. Require the `quality-ratchet/protected` commit status on `main`.
2. Create the `quality-ratchet-policy` environment.
3. Restrict that environment to protected branches and configure
   `@rylinjames` as its sole reviewer.

The manual workflow requires the exact reviewer allowlist before it can
authorize a commit. Missing, duplicate, team, or additional reviewers and a
weaker branch policy all fail closed because every configured reviewer could
otherwise grant deployment approval.

The protected quality policy forbids `tool.mypy.plugins`. Mypy imports plugin
code during analysis; permitting a repository-local plugin would let a pull
request replace that code and execute it inside the ordinary status-writing
job. Both the current protected policy and proposed policy updates are rejected
before Ruff or mypy starts if a plugin setting is present.

## Updating the gate or policy

Ordinary pull requests cannot authorize changes to the ratchet script,
workflows, CODEOWNERS, mutable baseline-like files, or the effective Ruff/mypy
sections of `pyproject.toml`.

For an intentional update:

1. Open the pull request and complete code-owner review.
2. From protected `main`, run **Quality ratchet protected policy update** with
   the pull-request number and reason.
3. Approve the `quality-ratchet-policy` environment deployment.
4. After the manual run succeeds, rerun the pull request's quality-ratchet job.

The manual workflow has three deliberately separate jobs. Its first job has no
write authority or Actions cache: it validates the exact current head SHA under
the old policy, parses the proposed Ruff/mypy configuration strictly as TOML
data without invoking either tool or importing configured plugins, statically
validates the proposed judge under protected policy, and uploads immutable
pre-execution evidence.

Its second job starts on a separate unprivileged runner only after that evidence
is sealed. Candidate tool configuration and tests can execute there, but the
job has no authority, produces no authorization outputs, and cannot mutate the
first runner or its immutable artifact.

Its third job starts on a fresh environment-protected runner. It checks out
only protected `main`, executes no candidate content, downloads only the exact
validation artifact, and re-queries both `refs/heads/main` and the current pull
request immediately before approval creation. Both must still identify the
sealed base and candidate. It then issues an immutable approval artifact plus a
status whose description binds the candidate SHA to the manual run ID, with a
second live-tip check immediately before that status write.

The ordinary protected workflow independently queries the current `main` tip
and pull request before comparison, rejecting stale event payloads and reruns
after `main` or the pull-request head advances. It repeats the live-tip check
after analysis and immediately before publishing the protected result. It also
queries the exact manual run and artifact through the GitHub API, downloads it
by artifact ID, and verifies its repository, base SHA, candidate SHA, workflow
path, and run ID with the protected judge. A status from an unrelated run, a
stale approval or event, or missing/expired evidence fails closed.
