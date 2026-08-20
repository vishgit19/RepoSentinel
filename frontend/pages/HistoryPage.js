import { html, useState, useEffect } from "../lib/html.js";
import { api } from "../lib/api.js";
import { Panel, Badge, Empty, ErrorBox } from "../components/ui.js";
import { statusTone, timeAgo, seconds, money, integer } from "../lib/format.js";

/** Past runs, reopenable because every trace is persisted. */
export function HistoryPage({ onOpenRun }) {
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState("");

  const refresh = async () => {
    try {
      setRuns(await api.listRuns(100));
    } catch (exception) {
      setError(exception.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  return html`
    <div class="page">
      <h2 class="section-title">Run history</h2>
      <p class="section-sub">
        Every run is persisted with its full timeline and trace, so a past
        investigation can be reopened and replayed exactly as it happened.
      </p>
      <${ErrorBox}>${error}<//>
      <${Panel} title="Runs" count=${runs.length} flush=${true}>
        ${runs.length === 0
          ? html`<${Empty}>No runs recorded yet.<//>`
          : html`<table class="grid">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Problem</th>
                  <th>Approach</th>
                  <th>Status</th>
                  <th class="num">Tests</th>
                  <th class="num">Tools</th>
                  <th class="num">Retries</th>
                  <th class="num">Latency</th>
                  <th class="num">Cost</th>
                  <th>When</th>
                </tr>
              </thead>
              <tbody>
                ${runs.map(
                  (run) => html`<tr
                    key=${run.run_id}
                    class="clickable"
                    onClick=${() => onOpenRun(run.run_id)}
                  >
                    <td class="mono">${run.run_id}</td>
                    <td>${run.benchmark_id || run.repo}</td>
                    <td>${run.strategy}</td>
                    <td>
                      <${Badge} tone=${statusTone(run.status)}>${run.status}<//>
                    </td>
                    <td class="num">
                      ${run.tests_failed
                        ? `${run.tests_passed}/${run.tests_passed + run.tests_failed}`
                        : integer(run.tests_passed)}
                    </td>
                    <td class="num">${integer(run.tool_calls)}</td>
                    <td class="num">${integer(run.retries)}</td>
                    <td class="num">${seconds(run.latency_ms)}</td>
                    <td class="num">${money(run.cost_usd)}</td>
                    <td>${timeAgo(run.created_at)}</td>
                  </tr>`
                )}
              </tbody>
            </table>`}
      <//>
    </div>
  `;
}
