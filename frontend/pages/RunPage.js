import { html, useState, useEffect, useMemo } from "../lib/html.js";
import { api } from "../lib/api.js";
import { Panel, Badge, Tabs, ErrorBox, Empty } from "../components/ui.js";
import { StepRail } from "../execution/StepRail.js";
import { Timeline } from "../execution/Timeline.js";
import { CodePanel, ChecksPanel } from "../execution/CodePanel.js";
import { MetricsPanel } from "../execution/MetricsPanel.js";
import { ReportPanel, ApprovalBar } from "../execution/ReportPanel.js";
import { useRunStream } from "../execution/useRunStream.js";
import { statusTone, integer } from "../lib/format.js";

const CUSTOM = "__custom__";

function InvestigationForm({
  benchmarks,
  models,
  strategies,
  defaults,
  onStart,
  onCanSubmit,
  busy,
  disabled,
}) {
  const [benchmarkId, setBenchmarkId] = useState("");
  const [repo, setRepo] = useState("");
  const [issue, setIssue] = useState("");
  const [issueId, setIssueId] = useState("");
  const [model, setModel] = useState("");
  const [strategy, setStrategy] = useState("agentic");
  const [memory, setMemory] = useState(true);

  // Selecting a benchmark fills in its issue text, which is what makes the
  // demo one click. Editing the text afterwards is still allowed.
  useEffect(() => {
    if (!benchmarks.length) return;
    if (!benchmarkId) {
      const first = benchmarks[0];
      setBenchmarkId(first.id);
      setIssue(first.issue);
      setIssueId(first.issue_id || "");
    }
  }, [benchmarks]);

  useEffect(() => {
    if (!model && defaults.model) setModel(defaults.model);
  }, [defaults.model]);

  const chooseBenchmark = (id) => {
    setBenchmarkId(id);
    if (id === CUSTOM) {
      setIssue("");
      setIssueId("");
      return;
    }
    const manifest = benchmarks.find((item) => item.id === id);
    if (manifest) {
      setIssue(manifest.issue);
      setIssueId(manifest.issue_id || "");
    }
  };

  const custom = benchmarkId === CUSTOM;
  const canSubmit =
    issue.trim().length > 0 && (custom ? repo.trim().length > 0 : Boolean(benchmarkId));

  useEffect(() => {
    if (onCanSubmit) onCanSubmit(canSubmit);
  }, [canSubmit, onCanSubmit]);

  const submit = (event) => {
    event.preventDefault();
    if (!canSubmit || busy || disabled) return;
    onStart({
      issue: issue.trim(),
      repo: custom ? repo.trim() : benchmarkId,
      benchmark_id: custom ? "" : benchmarkId,
      issue_id: issueId.trim(),
      model,
      strategy,
      memory_enabled: memory,
      auto_approve: false,
    });
  };

  const selected = benchmarks.find((item) => item.id === benchmarkId);

  return html`
    <${Panel} title="Investigation">
      <form id="investigation-form" onSubmit=${submit}>
        <label class="field">
          <span>Problem</span>
          <select value=${benchmarkId} onChange=${(e) => chooseBenchmark(e.target.value)}>
            ${benchmarks.map(
              (item) => html`<option key=${item.id} value=${item.id}>
                ${item.title}
              </option>`
            )}
            <option value=${CUSTOM}>Custom repository\u2026</option>
          </select>
          ${selected
            ? html`<div class="hint">
                ${selected.category} \u00b7 ${selected.difficulty}
                ${selected.expected_retry ? " \u00b7 expects a failed first patch" : ""}
                ${selected.expects_injection ? " \u00b7 contains a prompt injection" : ""}
              </div>`
            : null}
        </label>

        ${custom
          ? html`<label class="field">
              <span>Repository</span>
              <input
                type="text"
                placeholder="https://github.com/owner/repo or C:\\path\\to\\repo"
                value=${repo}
                onInput=${(e) => setRepo(e.target.value)}
              />
              <div class="hint">A local path or Git URL. It is copied, never modified.</div>
            </label>`
          : null}

        <label class="field">
          <span>Issue description</span>
          <textarea
            value=${issue}
            onInput=${(e) => setIssue(e.target.value)}
            placeholder="Users with an expired session token are still being authenticated."
          ></textarea>
        </label>

        <label class="field">
          <span>Issue / CVE id (optional)</span>
          <input type="text" value=${issueId} onInput=${(e) => setIssueId(e.target.value)} />
        </label>

        <label class="field">
          <span>Model</span>
          <select value=${model} onChange=${(e) => setModel(e.target.value)}>
            ${models.map(
              (item) => html`<option key=${item.id} value=${item.id} disabled=${!item.available}>
                ${item.label}${item.available ? "" : " (no credentials)"}
              </option>`
            )}
          </select>
        </label>

        <label class="field">
          <span>Approach</span>
          <select value=${strategy} onChange=${(e) => setStrategy(e.target.value)}>
            ${strategies.map(
              (item) => html`<option key=${item.id} value=${item.id}>
                ${item.baseline} \u2014 ${item.label}
              </option>`
            )}
          </select>
          ${strategies.find((item) => item.id === strategy)
            ? html`<div class="hint">
                ${strategies.find((item) => item.id === strategy).description}
              </div>`
            : null}
        </label>

        <label class="checkline">
          <input
            type="checkbox"
            checked=${memory}
            onChange=${(e) => setMemory(e.target.checked)}
          />
          Use repair memory from past runs
        </label>
        ${disabled
          ? html`<div class="hint">
              No model credentials are configured, so a run cannot start. Set
              OPENAI_API_KEY and restart the server.
            </div>`
          : null}
      </form>
    <//>
  `;
}

export function RunPage({ health, benchmarks, models, strategies, onOpenRun, openRunId }) {
  const { run, attach, load, reset, patch } = useRunStream();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState("code");
  const [deciding, setDeciding] = useState(false);
  const [canSubmit, setCanSubmit] = useState(false);

  useEffect(() => {
    if (openRunId && openRunId !== run.runId) {
      load(openRunId).catch((exception) => setError(exception.message));
    }
  }, [openRunId]);

  useEffect(() => {
    if (run.status !== "awaiting_approval") setDeciding(false);
  }, [run.status]);

  const start = async (request) => {
    setError("");
    setBusy(true);
    reset();
    try {
      const created = await api.startRun(request);
      attach(created.run_id, { steps: created.steps, status: created.status });
      onOpenRun(created.run_id);
    } catch (exception) {
      setError(exception.message);
    } finally {
      setBusy(false);
    }
  };

  const decide = async (approved, note) => {
    if (!run.runId || deciding) return;
    setDeciding(true);
    setError("");
    try {
      await api.decide(run.runId, approved, note);
      // Do not wait for the SSE status frame: a buffered stream used to leave
      // the approval bar up after the decision had already been recorded.
      patch({
        status: approved ? "approved" : "rejected",
        statusDetail: approved ? "approved by reviewer" : "rejected by reviewer",
      });
    } catch (exception) {
      setError(exception.message);
      setDeciding(false);
    }
  };

  const data = run.data || {};
  const checks = {
    targeted: data.targeted_tests,
    full: data.full_tests,
    security: data.security,
    lint: data.lint,
  };
  const running = run.streaming || run.status === "running";
  const noCredentials = health && !Object.values(health.providers || {}).some(Boolean);

  const steps = useMemo(
    () => (run.steps.length ? run.steps : []),
    [run.steps]
  );

  const tabs = [
    { id: "code", label: "Code" },
    { id: "checks", label: "Checks" },
    { id: "metrics", label: "Metrics" },
    { id: "report", label: "Report" },
  ];

  return html`
    <div class="main">
      <div class="col col-input">
        <div class="col-stack">
        <${InvestigationForm}
          benchmarks=${benchmarks}
          models=${models.models || []}
          strategies=${strategies}
          defaults=${{ model: models.default }}
          onStart=${start}
          onCanSubmit=${setCanSubmit}
          busy=${busy || running}
          disabled=${noCredentials}
        />
        <${ErrorBox}>${error}<//>
        ${steps.length ? html`<${StepRail} steps=${steps} counts=${run.stepCounts} />` : null}
        ${run.info
          ? html`<${Panel} title="Run environment">
              <dl class="kv">
                <dt>model</dt>
                <dd>${(run.info.model && run.info.model.model) || "\u2014"}</dd>
                <dt>sandbox</dt>
                <dd>${run.info.sandbox || "\u2014"}</dd>
                <dt>indexed</dt>
                <dd>
                  ${run.info.index
                    ? `${run.info.index.files} files, ${run.info.index.symbols} symbols`
                    : "\u2014"}
                </dd>
                <dt>chunks</dt>
                <dd>
                  ${run.info.retrieval ? integer(run.info.retrieval.chunks) : "\u2014"}
                </dd>
                <dt>memory</dt>
                <dd>
                  ${run.info.memory_enabled
                    ? `on (${integer(run.info.memory_records)} records)`
                    : "off"}
                </dd>
              </dl>
            <//>`
          : null}
        </div>
        <div class="col-footer">
          <button
            class="primary"
            type="submit"
            form="investigation-form"
            disabled=${busy || running || noCredentials || !canSubmit}
          >
            ${busy || running ? "Running\u2026" : "Run agent"}
          </button>
        </div>
      </div>

      <div class="col col-timeline">
        <div class="col-timeline-head">
          <div class="row">
            ${run.runId
              ? html`<${Badge} tone=${statusTone(run.status)}>${run.status}<//>`
              : null}
            ${run.runId ? html`<span class="mono faint">${run.runId}</span>` : null}
            <span class="grow"></span>
            ${run.statusDetail ? html`<span class="faint">${run.statusDetail}</span>` : null}
          </div>
          ${run.status === "awaiting_approval" && deciding
            ? html`<div class="approval-bar">
                <div class="msg">Decision recorded. Writing the report\u2026</div>
              </div>`
            : run.status === "awaiting_approval"
            ? html`<${ApprovalBar} onDecide=${decide} busy=${deciding} />`
            : null}
          <${ErrorBox}>${error}<//>
        </div>
        <div class="col-timeline-body">
          <${Timeline}
            events=${run.events}
            live=${running}
            tokens=${run.metrics.total_tokens}
            cost=${run.metrics.cost_usd}
          />
        </div>
      </div>

      <div class="col col-detail">
        <${Tabs} tabs=${tabs} active=${tab} onChange=${setTab} />
        ${tab === "code"
          ? html`<${CodePanel}
              inspected=${data.files_inspected}
              retrievedFiles=${data.retrieved_files}
              chunks=${data.retrieved_chunks}
              diff=${data.diff}
              patch=${data.patch}
            />`
          : null}
        ${tab === "checks" ? html`<${ChecksPanel} ...${checks} />` : null}
        ${tab === "metrics"
          ? html`<${MetricsPanel}
              metrics=${run.metrics}
              trace=${run.trace}
              checks=${checks}
              verified=${data.final_report ? data.final_report.verified : false}
              safety=${data.safety_events}
            />`
          : null}
        ${tab === "report" ? html`<${ReportPanel} report=${data.final_report} />` : null}
        ${!run.runId
          ? html`<${Panel} title="What you are about to watch">
              <${Empty}>
                The agent triages the issue, plans an investigation, retrieves
                only the code it asks for, calls tools, runs the tests, patches
                the defect and verifies the result \u2014 then asks you to approve.
              <//>
            <//>`
          : null}
      </div>
    </div>
  `;
}
