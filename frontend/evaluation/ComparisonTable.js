import { html } from "../lib/html.js";
import { Panel, Empty, Bar } from "../components/ui.js";
import { percent, ratio, money, seconds, integer } from "../lib/format.js";

/**
 * The headline comparison: every approach against the same problems.
 *
 * The full agent's row is highlighted, but nothing here is hard-coded in its
 * favour - each cell is an aggregate of recorded run outcomes.
 */
export function ComparisonTable({ comparison }) {
  if (!comparison || comparison.length === 0) {
    return html`<${Panel} title="Approach comparison">
      <${Empty}>Run an evaluation suite to populate this table.<//>
    <//>`;
  }

  return html`
    <${Panel} title="Approach comparison" flush=${true}>
      <table class="grid">
        <thead>
          <tr>
            <th>Approach</th>
            <th class="num">Cases</th>
            <th>Repair rate</th>
            <th class="num">Tests pass</th>
            <th class="num">Recovered</th>
            <th class="num">Gold recall</th>
            <th class="num">MRR</th>
            <th class="num">Tool calls</th>
            <th class="num">LLM calls</th>
            <th class="num">Latency</th>
            <th class="num">Cost</th>
          </tr>
        </thead>
        <tbody>
          ${comparison.map(
            (row) => html`<tr key=${row.strategy} class=${row.strategy === "agentic" ? "highlight" : ""}>
              <td>
                <b>${row.baseline}</b> \u00b7 ${row.label}
              </td>
              <td class="num">${row.cases}</td>
              <td>
                <div class="row" style=${{ gap: "7px" }}>
                  <${Bar} value=${row.repair_rate} tone=${row.repair_rate >= 0.5 ? "good" : "bad"} />
                  <span class="mono">${percent(row.repair_rate)}</span>
                </div>
              </td>
              <td class="num">${percent(row.full_pass_rate)}</td>
              <td class="num">${percent(row.recovery_rate)}</td>
              <td class="num">${ratio(row.avg_gold_recall)}</td>
              <td class="num">${ratio(row.avg_mrr)}</td>
              <td class="num">${ratio(row.avg_tool_calls)}</td>
              <td class="num">${ratio(row.avg_llm_calls)}</td>
              <td class="num">${seconds(row.avg_latency_ms)}</td>
              <td class="num">${money(row.avg_cost_usd)}</td>
            </tr>`
          )}
        </tbody>
      </table>
    <//>
  `;
}

export function RetrievalTable({ comparison }) {
  if (!comparison || comparison.length === 0) return null;
  return html`
    <${Panel} title="Retrieval quality" flush=${true}>
      <table class="grid">
        <thead>
          <tr>
            <th>Approach</th>
            <th class="num">Recall@K</th>
            <th class="num">Precision@K</th>
            <th class="num">MRR</th>
            <th class="num">Gold file recall</th>
            <th class="num">Correct file patched</th>
          </tr>
        </thead>
        <tbody>
          ${comparison.map(
            (row) => html`<tr key=${row.strategy} class=${row.strategy === "agentic" ? "highlight" : ""}>
              <td><b>${row.baseline}</b> \u00b7 ${row.label}</td>
              <td class="num">${ratio(row.avg_recall_at_k)}</td>
              <td class="num">${ratio(row.avg_precision_at_k)}</td>
              <td class="num">${ratio(row.avg_mrr)}</td>
              <td class="num">${ratio(row.avg_gold_recall)}</td>
              <td class="num">${percent(row.correct_file_rate)}</td>
            </tr>`
          )}
        </tbody>
      </table>
    <//>
  `;
}

export function SafetyTable({ comparison }) {
  if (!comparison || comparison.length === 0) return null;
  return html`
    <${Panel} title="Safety and efficiency" flush=${true}>
      <table class="grid">
        <thead>
          <tr>
            <th>Approach</th>
            <th class="num">Blocked commands</th>
            <th class="num">Injections detected</th>
            <th class="num">Unnecessary tool calls</th>
            <th class="num">Retries</th>
            <th class="num">Regressions</th>
            <th class="num">Tokens</th>
          </tr>
        </thead>
        <tbody>
          ${comparison.map(
            (row) => html`<tr key=${row.strategy} class=${row.strategy === "agentic" ? "highlight" : ""}>
              <td><b>${row.baseline}</b> \u00b7 ${row.label}</td>
              <td class="num">${integer(row.blocked_commands)}</td>
              <td class="num">${integer(row.injections_detected)}</td>
              <td class="num">${ratio(row.avg_unnecessary_tool_calls)}</td>
              <td class="num">${ratio(row.avg_retries)}</td>
              <td class="num">${percent(row.regression_rate)}</td>
              <td class="num">${integer(Math.round(row.avg_tokens))}</td>
            </tr>`
          )}
        </tbody>
      </table>
    <//>
  `;
}
