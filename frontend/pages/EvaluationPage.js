import { html, useState, useEffect, useRef } from "../lib/html.js";
import { api, subscribe } from "../lib/api.js";
import { Panel, Badge, ErrorBox, Empty } from "../components/ui.js";
import {
  ComparisonTable,
  RetrievalTable,
  SafetyTable,
} from "../evaluation/ComparisonTable.js";
import { TrajectoryCompare } from "../evaluation/TrajectoryCompare.js";
import { timeAgo } from "../lib/format.js";

const SUITE_EVENTS = [
  "suite_started",
  "case_started",
  "case_finished",
  "suite_finished",
  "stream_end",
  "heartbeat",
];

function SuiteForm({ benchmarks, strategies, models, onStart, busy }) {
  const [chosenBenchmarks, setChosenBenchmarks] = useState([]);
  const [chosenStrategies, setChosenStrategies] = useState([]);
  const [model, setModel] = useState("");

  useEffect(() => {
    if (benchmarks.length && chosenBenchmarks.length === 0) {
      setChosenBenchmarks([benchmarks[0].id]);
    }
  }, [benchmarks]);

  useEffect(() => {
    if (strategies.length && chosenStrategies.length === 0) {
      setChosenStrategies(strategies.map((item) => item.id));
    }
  }, [strategies]);

  useEffect(() => {
    if (!model && models.default) setModel(models.default);
  }, [models.default]);

  const toggle = (list, setList, value) =>
    setList(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);

  const cases = chosenBenchmarks.length * chosenStrategies.length;

  return html`
    <${Panel} title="Run a comparison">
      <div class="row" style=${{ alignItems: "flex-start", gap: "24px" }}>
        <div style=${{ minWidth: "240px" }}>
          <div class="k" style=${{ marginBottom: "6px", fontSize: "11px" }}>Problems</div>
          ${benchmarks.map(
            (item) => html`<label class="checkline" key=${item.id}>
              <input
                type="checkbox"
                checked=${chosenBenchmarks.includes(item.id)}
                onChange=${() => toggle(chosenBenchmarks, setChosenBenchmarks, item.id)}
              />
              ${item.title}
            </label>`
          )}
        </div>
        <div style=${{ minWidth: "260px" }}>
          <div class="k" style=${{ marginBottom: "6px", fontSize: "11px" }}>Approaches</div>
          ${strategies.map(
            (item) => html`<label class="checkline" key=${item.id}>
              <input
                type="checkbox"
                checked=${chosenStrategies.includes(item.id)}
                onChange=${() => toggle(chosenStrategies, setChosenStrategies, item.id)}
              />
              ${item.baseline} \u2014 ${item.label}
            </label>`
          )}
        </div>
        <div style=${{ minWidth: "200px" }}>
          <label class="field">
            <span>Model (identical for every case)</span>
            <select value=${model} onChange=${(e) => setModel(e.target.value)}>
              ${(models.models || []).map(
                (item) => html`<option key=${item.id} value=${item.id} disabled=${!item.available}>
                  ${item.label}
                </option>`
              )}
            </select>
          </label>
          <button
            class="primary"
            disabled=${busy || cases === 0}
            onClick=${() =>
              onStart({
                benchmark_ids: chosenBenchmarks,
                strategies: chosenStrategies,
                model,
              })}
          >
            ${busy ? "Suite running\u2026" : `Run ${cases} case(s)`}
          </button>
          <div class="hint">
            Each case is a full agent run against a fresh copy of the repository.
            Expect roughly a minute per case.
          </div>
        </div>
      </div>
    <//>
  `;
}

export function EvaluationPage({ benchmarks, strategies, models }) {
  const [suites, setSuites] = useState([]);
  const [suiteId, setSuiteId] = useState("");
  const [summary, setSummary] = useState(null);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [focus, setFocus] = useState("");
  const closeRef = useRef(null);

  const refreshSuites = async () => {
    try {
      const payload = await api.listEvaluations();
      setSuites(payload.suites || []);
      return payload;
    } catch (exception) {
      setError(exception.message);
      return null;
    }
  };

  useEffect(() => {
    refreshSuites();
    return () => closeRef.current && closeRef.current();
  }, []);

  const openSuite = async (id) => {
    setSuiteId(id);
    setError("");
    try {
      const payload = await api.getEvaluation(id);
      setSummary(payload);
      setProgress(payload.progress);
      if (payload.benchmarks && payload.benchmarks.length) {
        setFocus((current) => current || payload.benchmarks[0]);
      }
    } catch (exception) {
      setError(exception.message);
    }
  };

  const watch = (id) => {
    if (closeRef.current) closeRef.current();
    closeRef.current = subscribe(
      `/api/evaluations/${id}/events`,
      SUITE_EVENTS,
      async (event) => {
        if (event.type === "case_started" || event.type === "suite_started") {
          setProgress({
            completed: event.completed || 0,
            total: event.total || 0,
            current: `${event.benchmark_id || ""} ${event.strategy || ""}`.trim(),
            finished: false,
          });
        }
        if (event.type === "case_finished") {
          setProgress({
            completed: event.completed,
            total: event.total,
            current: "",
            finished: false,
          });
          await openSuite(id);
        }
        if (event.type === "suite_finished") {
          setProgress({ completed: event.completed, total: event.total, finished: true });
          setBusy(false);
          await openSuite(id);
          await refreshSuites();
        }
      },
      () => setBusy(false)
    );
  };

  const start = async (request) => {
    setError("");
    setBusy(true);
    setSummary(null);
    try {
      const created = await api.startEvaluation(request);
      setSuiteId(created.suite_id);
      watch(created.suite_id);
    } catch (exception) {
      setError(exception.message);
      setBusy(false);
    }
  };

  const comparison = summary ? summary.comparison : [];
  const cases = summary ? summary.cases : [];
  const problems = summary ? summary.benchmarks : [];

  return html`
    <div class="page">
      <h2 class="section-title">Evaluation</h2>
      <p class="section-sub">
        The same problem, executed by five configurations of the same system.
        Baselines A\u2013D differ only in how they gather context and whether they
        may retry; every metric below is computed from recorded test output and
        applied diffs, not from what a model claimed about its own work.
      </p>

      <div class="stack">
        <${SuiteForm}
          benchmarks=${benchmarks}
          strategies=${strategies}
          models=${models}
          onStart=${start}
          busy=${busy}
        />
        <${ErrorBox}>${error}<//>

        ${progress && !progress.finished
          ? html`<${Panel} title="Suite progress">
              <div class="row">
                <${Badge} tone="info">
                  ${progress.completed}/${progress.total} cases
                <//>
                <span class="dim">${progress.current || "starting\u2026"}</span>
              </div>
            <//>`
          : null}

        ${suites.length
          ? html`<${Panel} title="Recorded suites" flush=${true}>
              <table class="grid">
                <thead>
                  <tr>
                    <th>Suite</th>
                    <th class="num">Results</th>
                    <th>When</th>
                  </tr>
                </thead>
                <tbody>
                  ${suites.map(
                    (suite) => html`<tr
                      key=${suite.suite_id}
                      class=${`clickable ${suite.suite_id === suiteId ? "highlight" : ""}`}
                      onClick=${() => openSuite(suite.suite_id)}
                    >
                      <td class="mono">${suite.suite_id}</td>
                      <td class="num">${suite.results}</td>
                      <td>${timeAgo(suite.created_at)}</td>
                    </tr>`
                  )}
                </tbody>
              </table>
            <//>`
          : null}

        ${summary
          ? html`
              <${ComparisonTable} comparison=${comparison} />
              <${RetrievalTable} comparison=${comparison} />
              <${SafetyTable} comparison=${comparison} />
              ${problems.length
                ? html`<div class="row">
                    <span class="k faint">Problem</span>
                    ${problems.map(
                      (problem) => html`<button
                        key=${problem}
                        class="ghost"
                        style=${problem === focus
                          ? { borderColor: "var(--accent)", color: "var(--text)" }
                          : {}}
                        onClick=${() => setFocus(problem)}
                      >
                        ${problem}
                      </button>`
                    )}
                  </div>`
                : null}
              <${TrajectoryCompare} benchmarkId=${focus} cases=${cases} />
            `
          : suites.length === 0
          ? html`<${Panel} title="Approach comparison">
              <${Empty}>No suites recorded yet. Run a comparison above.<//>
            <//>`
          : null}
      </div>
    </div>
  `;
}
