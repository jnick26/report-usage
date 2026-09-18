# Harness Usage

A local browser application for coding-agent usage, published as `report-usage`.
Imports Pi, Codex, Claude Code, Copilot in VS Code, and Copilot CLI as independent
sources into one DuckDB database. Missing usage is unavailable, not zero.

## Install with uv

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and
access to this repository over HTTPS or SSH. The supported desktop platform is macOS;
Python 3.13 is required. Windows is not supported by the current runtime.

```sh
uv tool install --python 3.13 'git+https://github.com/jnick26/report-usage.git@v0.1.1'
harness-usage
```

SSH works too, if you prefer SSH-key authentication:

```sh
uv tool install --python 3.13 'git+ssh://git@github.com/jnick26/report-usage.git@v0.1.1'
```

For a private repository, authenticate Git for the chosen transport first.
HTTPS does not require an SSH key; do not put access tokens in the install URL.

The command opens `http://127.0.0.1:8765`. If the command is not on your PATH,
run `uv tool update-shell` and restart your shell. The native macOS app bundle
is not needed for uv installation.

The tagged install stays pinned. To switch to a later release, install its tag
with `uv tool install --force --python 3.13 <git-url-with-new-tag>`.

## Configure sources

On first launch, existing standard history directories are detected automatically:

| Source | Typical macOS location |
| --- | --- |
| Pi | `~/.pi/agent/sessions` |
| Codex | `~/.codex/sessions`, `~/.codex/archived_sessions` |
| Claude Code | `~/.claude/projects` |
| Copilot in VS Code | `~/Library/Application Support/Code/User/workspaceStorage` |
| Copilot in VS Code Insiders | `~/Library/Application Support/Code - Insiders/User/workspaceStorage` |
| Copilot CLI | `~/.copilot` (includes `session-state` and optional `session-store.db`) |

Missing directories are skipped. Saved source settings (including an empty list)
are preserved; explicit `--root` arguments replace them. You can inspect or edit
the selected directories in **Sources**. If no directories are found, detection
is retried on the next launch until you save a configuration.

Home overrides are respected: `PI_CODING_AGENT_DIR`, `CODEX_HOME`,
`CLAUDE_CONFIG_DIR`, and `COPILOT_HOME`. Custom VS Code user-data locations
and other nonstandard paths can be added in **Sources** or with `--root`.
Pi's `PI_CODING_AGENT_SESSION_DIR`, when set, takes precedence over its agent home.
Imports run on launch and when you choose **Refresh**. Original history files
are read, not modified.

Existing Copilot configurations pointing only at `session-state` stay unchanged.
To include the sibling `session-store.db`, add its parent Copilot home directory
in **Sources**. The importer never expands a saved root to read outside it.

```sh
harness-usage --port 8766 --timezone Europe/Kyiv
harness-usage --root /absolute/path/to/history --no-browser
harness-usage --data-dir /absolute/path/to/app-data
```

App data defaults to `~/Library/Application Support/Harness Usage`; tool upgrades
do not replace it. Only one running instance may use a given data directory.
The server binds to loopback. Do not expose it publicly: it is a local tool,
not an authenticated multi-user service. Installation downloads dependencies;
normal usage and bundled pricing work offline.

## Accounting limits

- Pi and Codex use their source-specific token and reconciliation rules.
- Claude output requires a valid per-record writer version of at least 2.1.97
  and a valid final output counter. Older or unversioned output stays unavailable.
- VS Code schema-v3 JSON/JSONL supports core usage/model totals and exact retained
  `result.usage.promptTokens/completionTokens` or
  `result.metadata.promptTokens/outputTokens` pairs. The latter pairs describe
  one call: output is a lower bound (`≥`), not an exact whole-turn total. Gross
  prompt evidence does not establish a fresh/cache split. Core fields take
  precedence; equal overlaps count once and conflicting valid pairs are unresolved.
- Copilot CLI supports the qualified durable event format. Workspace-only older
  history has unavailable usage, not zero usage.
  The optional schema-8 `session-store.db` is read as a consistent, read-only
  snapshot, including committed WAL records. Retained database calls are lower
  bounds and are reconciled with overlapping event summaries, never added blindly.
  Conflicting or unproven overlaps remain unresolved. Cache-inclusive input does
  not establish fresh input when cache usage is present.
- AI credits, nano-AIU, premium requests, request counts, recorded USD estimates,
  and API-equivalent cost estimates stay separate. None establishes your bill or
  subscription allocation. Selected model names alone are not priced.
- Copilot VS Code and CLI have no general exact cross-source request join; their
  combined usage is not a deduplicated billing total.
- Transcripts require supported retained source files. Deleted history cannot be
  reconstructed. Copilot CLI prefers native event transcripts; database-only
  sessions can show retained turn summaries, explicitly labeled lossy. These
  summaries do not reconstruct tool calls, reasoning, or branches, and are read
  only on demand, never stored in the usage ledger. Missing turns remain
  unavailable. Future per-call capture is not implemented or enabled.

Pricing uses a bundled [models.dev](https://models.dev) snapshot; its license is
included in the package. Unknown models or insufficient usage remain unpriced.
Direct catalog matches take precedence. Claude Bridge uses Anthropic reference
prices; Copilot models missing a direct match use an unambiguous official-provider
match, including supported Claude spelling aliases. These are API-equivalent
estimates, not billed charges; recorded model/provider identities stay unchanged.
Models with no recorded provider, including Claude Code records, can also use
an unambiguous official-provider reference match.

## Development

```sh
uv sync --locked
uv run pytest
uv run mypy src/harness_usage
```

The fixtures are synthetic. Local histories, databases, credentials, research
results and development recovery archives are excluded from this repository.
Historical-reader migration checks require the unpublished development snapshots;
those checks skip explicitly when the archive is absent, while normal tests run.
Packaged-runtime tests also require `HARNESS_USAGE_BUNDLE` to point to a locally
built executable. Skipped checks are not a claim of release-package validation.

Optional macOS bundle build: `uv run pyinstaller harness-usage.spec --noconfirm`.
