import { html, useState, useEffect, ReactDOM, React } from "./lib/html.js";
import { api } from "./lib/api.js";
import { Chip, ErrorBox } from "./components/ui.js";
import { RunPage } from "./pages/RunPage.js";
import { EvaluationPage } from "./pages/EvaluationPage.js";
import { HistoryPage } from "./pages/HistoryPage.js";
import { integer } from "./lib/format.js";

const TABS = [
  { id: "run", label: "Run agent" },
  { id: "evaluation", label: "Evaluation" },
  { id: "history", label: "History" },
];

function SystemChips({ health }) {
  if (!health) return null;
  const sandbox = health.sandbox || {};
  const providers = Object.entries(health.providers || {}).filter(([, ok]) => ok);
  return html`
    <div class="sysinfo">
      <${Chip}
        tone=${sandbox.backend === "docker" ? "ok" : "warn"}
        label="sandbox"
        value=${sandbox.backend || "?"}
      />
      <${Chip} tone="info" label="security" value=${health.security_backend} />
      <${Chip} tone="info" label="vectors" value=${health.vector_store} />
      <${Chip} tone="info" label="tools" value=${integer(health.tools)} />
      <${Chip} tone="info" label="mcp" value=${integer(health.mcp_tools)} />
      <${Chip}
        tone=${health.memory_records ? "ok" : ""}
        label="memory"
        value=${integer(health.memory_records)}
      />
      <${Chip}
        tone=${providers.length ? "ok" : "fail"}
        label="llm"
        value=${providers.length ? providers.map(([name]) => name).join(",") : "none"}
      />
    </div>
  `;
}

function App() {
  const [tab, setTab] = useState("run");
  const [health, setHealth] = useState(null);
  const [benchmarks, setBenchmarks] = useState([]);
  const [models, setModels] = useState({ default: "", models: [] });
  const [strategies, setStrategies] = useState([]);
  const [openRunId, setOpenRunId] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.health(), api.benchmarks(), api.models(), api.strategies()])
      .then(([healthPayload, benchmarkPayload, modelPayload, strategyPayload]) => {
        setHealth(healthPayload);
        setBenchmarks(benchmarkPayload);
        setModels(modelPayload);
        setStrategies(strategyPayload);
      })
      .catch((exception) => setError(exception.message));
  }, []);

  const openRun = (runId) => {
    setOpenRunId(runId);
    setTab("run");
  };

  return html`
    <div class="app">
      <div class="topbar">
        <div class="brand">
          <img class="mark" src="/static/logo.svg" width="22" height="22" alt="" />
          <span>RepoSentinel</span>
          <span class="tagline">agentic secure code repair and verification</span>
        </div>
        <nav class="nav">
          ${TABS.map(
            (item) => html`<button
              key=${item.id}
              class=${item.id === tab ? "active" : ""}
              onClick=${() => setTab(item.id)}
            >
              ${item.label}
            </button>`
          )}
        </nav>
        <span class="spacer"></span>
        <${SystemChips} health=${health} />
      </div>

      ${error ? html`<div style=${{ padding: "16px" }}><${ErrorBox}>${error}<//></div>` : null}

      ${tab === "run"
        ? html`<${RunPage}
            health=${health}
            benchmarks=${benchmarks}
            models=${models}
            strategies=${strategies}
            onOpenRun=${setOpenRunId}
            openRunId=${openRunId}
          />`
        : null}
      ${tab === "evaluation"
        ? html`<${EvaluationPage}
            benchmarks=${benchmarks}
            strategies=${strategies}
            models=${models}
          />`
        : null}
      ${tab === "history" ? html`<${HistoryPage} onOpenRun=${openRun} />` : null}
    </div>
  `;
}

ReactDOM.createRoot(document.getElementById("root")).render(
  React.createElement(App)
);
