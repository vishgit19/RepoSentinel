<p align="center">
  <img src="docs/brand/logo.png" width="128" alt="RepoSentinel mark" />
</p>

<h1 align="center">RepoSentinel</h1>

<p align="center">
  <strong>Agentic secure code repair and verification</strong>
</p>

<p align="center">
  An agent that inspects a real repository, proposes a patch, runs the tests
  and a security scan, and only then asks a human to approve.
</p>

<p align="center">
  <a href="https://github.com/vishgit19/RepoSentinel/actions/workflows/ci.yml"><img src="https://github.com/vishgit19/RepoSentinel/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-58a6ff?style=flat-square" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/license-MIT-3fb950?style=flat-square" alt="MIT License" />
  <img src="https://img.shields.io/badge/human--in--the--loop-required-d29922?style=flat-square" alt="Human in the loop" />
</p>

<p align="center">
  <img src="docs/brand/ui.png" width="960" alt="RepoSentinel run UI with live timeline, checks, and human approval" />
</p>

This is a working system, not a mock. The agent shells out to pytest and ruff,
retrieves code with BM25 + embeddings + a symbol graph, and streams every
node, tool call and check to a browser UI. When a model is not configured the
UI still loads; a run will not start.

If the first patch fails, it diagnoses the failure and tries again.

## What you can watch it do

Five bundled problems, each a small defective Python repository with failing
tests:

| Id | What is wrong | What it is for |
| --- | --- | --- |
| `logic_bug` | `SessionToken.is_expired` adds TTL twice | Single-file logic error |
| `cache_bug` | Cache keys drop the locale, so German readers get English text | Cross-file: the symptom is in the service, the defect is in `make_key` |
| `sql_injection` | Email lookup interpolates into SQL | Vulnerability the built-in scanner flags; engine already supports `?` |
| `retry_bug` | Retries 404s and never backs off | Two independent defects; a first patch that only tightens 4xx still fails |
| `injection` | Off-by-one discount, plus a prompt injection in the module docstring | Guardrails must detect the injection, ignore it, and still fix the code |

The UI at `/` has three pages:

- **Run agent** — live timeline of every node, tool call, retrieved chunk,
  generated diff, test result and security finding. A human approval bar
  appears before anything is treated as accepted. Nothing is pushed to a
  remote repository from a benchmark run.
- **Evaluation** — the same problem executed as baselines A–D and the full
  agent E, with retrieval / repair / safety metrics computed from recorded
  test output and applied diffs, not from what the model claimed.
- **History** — every past run, reopenable, because the timeline and trace
  are persisted in SQLite.

## Running it

Python 3.11+ (3.12 is what this repo was developed on). No Node toolchain is
required: the frontend is React served as static files with vendored UMD
builds.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e .
cp .env.example .env   # then put OPENAI_API_KEY in it, or export it
python scripts/serve.py --port 8000
```

Open <http://127.0.0.1:8000>. Pick a problem, click **Run agent**.

A one-shot CLI run, useful when you do not want the UI:

```bash
python scripts/run_agent.py --benchmark logic_bug --auto-approve
```

### Docker

```bash
docker compose up --build
```

The image uses the local subprocess sandbox (there is no Docker-in-Docker).
Pass `OPENAI_API_KEY` in the environment. Data lives in the `sentinel-data`
volume.

### MCP

The same read-only repository tools the agent uses are exposed over MCP stdio:

```bash
python -m reposentinel.mcp_server --repo logic_bug
```

Point an MCP client at that command, with `cwd` set to this checkout.
`apply_patch` and `write_file` are registered internally but **not** exposed:
a remote client cannot edit code through this server. Tools run against an
isolated copy of the repository, not your working tree.

## How a run is organised

```
INPUT → TRIAGE → MEMORY → PLANNER → RETRIEVE → TOOLS → ROOT CAUSE
      → PATCH → CHECKS ┬─(pass)→ VERIFIER → APPROVAL → REPORT
                       └─(fail)→ REFLECT ─(retry)→ PATCH
                                      └─(give up)→ REPORT
```

Baselines A–D share this graph. They differ in three flags only:

| Baseline | Id | Retrieval | Tools | Retry loop |
| --- | --- | --- | --- | --- |
| A | `llm_only` | seed files only | no | no |
| B | `vector_rag` | dense | no | no |
| C | `hybrid_rag` | BM25 + dense + rerank | no | no |
| D | `graph_rag` | hybrid + symbol-graph expansion | no | no |
| E | `agentic` | graph | yes | yes |

A baseline that could retry would erase the comparison. `retry_bug` exists
specifically so E can be seen recovering from a failed first patch.

## Architecture

```
frontend/                 build-less React UI (vendored react + htm)
backend/reposentinel/
  api/app.py              FastAPI + SSE
  graph/                  LangGraph workflow and nodes
  tools/                  repository, execution, retrieval tools
  retrieval/              BM25, embeddings, vector store, rerank, graph
  sandbox/                local subprocess + optional Docker, with guardrails
  mcp_server/             JSON-RPC 2.0 stdio adapter over the same tools
  evaluation/             metrics + harness for baselines A–E
  observability/          timeline emitter, event bus, SQLite store, tracer
  memory/                 past-repair recall
  models/providers/       OpenAI-compatible + scripted (tests)
benchmarks/<id>/          manifest.json + repo/ with a seeded defect
```

**Sandbox.** `REPOSENTINEL_SANDBOX_BACKEND=auto` uses Docker when `docker` is
on PATH, otherwise a local subprocess with an allow-list, a timeout, a
scrubbed environment and a workspace chroot. Commands like `rm`, `curl`,
`pip` and `python -c` are refused. Flag values such as `--output-format=json`
are not mistaken for the `format` program.

**Retrieval.** Each run copies the repository, builds an AST index, chunks on
symbol boundaries, and retrieves with BM25 + dense vectors (OpenAI embeddings
when a key is present, otherwise a hashing fallback). Results are fused,
reranked, and expanded one hop along `calls` / `contains` / `imports`.

**Guardrails.** Path escape is refused. Secrets in file contents are redacted.
Prompt-injection patterns in retrieved or read files are recorded as safety
events and wrapped as untrusted data before they enter a prompt. The
`injection` benchmark plants several of those patterns in a docstring; the
correct behaviour is to ignore them and still repair the off-by-one.

**Verification.** A model cannot declare success. Deterministic gates (targeted
tests, full suite, lint, security scan) run on the real files; the model is
only asked afterwards whether behaviour looks preserved. Human approval is a
real blocking wait on the graph thread.

**GitHub.** A pull request is attempted only after a human approves, and only
when `REPOSENTINEL_ALLOW_GITHUB_PUSH` is set, a `GITHUB_TOKEN` is present, and
the workspace has a GitHub remote. Benchmark runs have no remote; the timeline
records that no PR was opened.

## Tests

```bash
python -m pytest tests -q
```

That suite covers guardrails, the sandbox, workspace git isolation, tools,
retrieval, the HTTP API, the MCP stdio server, evaluation metrics, and the
claim that each benchmark's named tests actually fail in the seeded tree.
It does not call a paid model. End-to-end agent behaviour is
`scripts/run_agent.py` or the UI.

## Configuration

Environment variables use the `REPOSENTINEL_` prefix. Conventional names such
as `OPENAI_API_KEY`, `GITHUB_TOKEN` and `DATABASE_URL` are also honoured; the
prefixed form wins when both are set. See `.env.example`.

Optional extras (`requirements-optional.txt`): `psycopg` for pgvector,
`semgrep` for the Semgrep security backend. Without them the system uses
SQLite + NumPy vector search and a built-in AST scanner. Those fallbacks are
real implementations, not stubs.

## License

[MIT](LICENSE)
