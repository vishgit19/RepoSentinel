import { html, useState } from "../lib/html.js";
import { Panel, Empty, Badge } from "../components/ui.js";

export function ApprovalBar({ onDecide, busy }) {
  const [note, setNote] = useState("");
  return html`
    <div class="approval-bar">
      <div class="msg">
        <b>Human approval required.</b> The patch is in the sandbox only. Nothing is
        pushed to a remote repository without this decision.
        <input
          type="text"
          placeholder="Optional note for the record"
          value=${note}
          onInput=${(e) => setNote(e.target.value)}
          style=${{ marginTop: "7px" }}
        />
      </div>
      <button class="approve" type="button" disabled=${busy} onClick=${() => onDecide(true, note)}>
        ${busy ? "Recording\u2026" : "Approve"}
      </button>
      <button class="reject" type="button" disabled=${busy} onClick=${() => onDecide(false, note)}>
        Reject
      </button>
    </div>
  `;
}

export function ReportPanel({ report }) {
  if (!report) {
    return html`<${Panel} title="Final report">
      <${Empty}>The report is written after verification.<//>
    <//>`;
  }
  return html`
    <${Panel} title="Final report">
      <div class="report">
        <div class="row" style=${{ marginBottom: "13px" }}>
          <${Badge} tone=${report.verified ? "ok" : "fail"}>
            ${report.verified ? "Solution verified" : "Not verified"}
          <//>
        </div>

        <section>
          <h4>Problem</h4>
          <p>${report.problem}</p>
        </section>

        <section>
          <h4>Root cause</h4>
          <p>${report.root_cause}</p>
        </section>

        <section>
          <h4>Changed files</h4>
          ${report.changed_files && report.changed_files.length
            ? html`<p>
                ${report.changed_files.map(
                  (file) => html`<code key=${file}>${file}</code> `
                )}
              </p>`
            : html`<p class="faint">Nothing was changed.</p>`}
        </section>

        <section>
          <h4>Explanation</h4>
          <p>${report.explanation}</p>
        </section>

        ${report.validation_performed && report.validation_performed.length
          ? html`<section>
              <h4>Validation performed</h4>
              <ul>
                ${report.validation_performed.map(
                  (item, index) => html`<li key=${index}>${item}</li>`
                )}
              </ul>
            </section>`
          : null}

        ${report.evidence && report.evidence.length
          ? html`<section>
              <h4>Evidence</h4>
              <ul>
                ${report.evidence.slice(0, 12).map(
                  (item, index) => html`<li key=${index}><code>${item}</code></li>`
                )}
              </ul>
            </section>`
          : null}

        ${report.remaining_risks && report.remaining_risks.length
          ? html`<section>
              <h4>Remaining risks</h4>
              <ul>
                ${report.remaining_risks.map(
                  (item, index) => html`<li key=${index}>${item}</li>`
                )}
              </ul>
            </section>`
          : null}
      </div>
    <//>
  `;
}
