import { html } from "../lib/html.js";
import { Panel } from "../components/ui.js";
import { STATUS_GLYPH, glyphClass } from "../lib/format.js";

/**
 * The fixed vertical workflow rail.
 *
 * Nodes can execute more than once (patch and checks repeat on every retry),
 * so repeats collapse into a single row with a run count rather than growing
 * the rail and losing the shape of the workflow.
 */
export function StepRail({ steps, counts }) {
  return html`
    <${Panel} title="Workflow" flush=${true}>
      <div class="rail">
        ${steps.map((step) => {
          const status = step.status || "pending";
          const runs = counts[step.node] || 0;
          return html`<div key=${step.node} class=${`rail-step ${status}`}>
            <span class=${`glyph ${glyphClass(status)} ${status === "running" ? "pulse" : ""}`}>
              ${STATUS_GLYPH[status] || STATUS_GLYPH.pending}
            </span>
            <span>${step.label}</span>
            ${runs > 1 ? html`<span class="repeat">x${runs}</span>` : null}
          </div>`;
        })}
      </div>
    <//>
  `;
}
