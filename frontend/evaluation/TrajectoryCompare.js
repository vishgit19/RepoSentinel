import { html } from "../lib/html.js";
import { Panel, Empty, Badge } from "../components/ui.js";
import { STATUS_GLYPH, glyphClass, money, seconds } from "../lib/format.js";

/**
 * Side-by-side trajectories for one problem.
 *
 * This is the clearest single view of what the agent adds: a baseline stops at
 * its first failed patch, while the full agent diagnoses the failure, pulls in
 * more context and tries again.
 */
export function TrajectoryCompare({ benchmarkId, cases }) {
  const relevant = (cases || []).filter((item) => item.benchmark_id === benchmarkId);
  if (relevant.length === 0) {
    return html`<${Panel} title="Trajectories">
      <${Empty}>Select a problem with recorded results.<//>
    <//>`;
  }

  relevant.sort((a, b) => (a.baseline || "").localeCompare(b.baseline || ""));

  return html`
    <${Panel} title=${`Trajectories \u00b7 ${benchmarkId}`}>
      <div class="traj-grid">
        ${relevant.map((item) => {
          const verified = item.repair && item.repair.verified;
          const recovered = item.agent && item.agent.recovered_after_failure;
          return html`<div class="traj" key=${item.strategy}>
            <header>
              <span class="name">${item.baseline} \u00b7 ${item.strategy_label}</span>
              <${Badge} tone=${verified ? "ok" : "fail"}>
                ${verified ? "repaired" : "not repaired"}
              <//>
            </header>
            <ol>
              ${(item.trajectory || []).map(
                (step, index) => html`<li key=${index}>
                  <span class=${`g ${glyphClass(step.status)}`}>
                    ${STATUS_GLYPH[step.status] || STATUS_GLYPH.pending}
                  </span>
                  <span class="n">${step.node}</span>
                  <span>${step.title}</span>
                </li>`
              )}
            </ol>
            <div class="stop">
              ${recovered ? "recovered after a failed patch \u00b7 " : ""}
              ${item.agent ? `${item.agent.retries} retries` : ""} \u00b7
              ${item.agent ? seconds(item.agent.latency_ms) : ""} \u00b7
              ${item.agent ? money(item.agent.cost_usd) : ""}
            </div>
          </div>`;
        })}
      </div>
    <//>
  `;
}
