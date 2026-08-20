import { html, useState } from "../lib/html.js";
import { Panel, Empty, Badge } from "../components/ui.js";
import { Diff, diffStat } from "../components/Diff.js";

/** Where a chunk came from, and which retriever kept it. */
function provenanceLabel(provenance) {
  if (!provenance) return "";
  const parts = [];
  if (provenance.retriever) parts.push(provenance.retriever);
  if (provenance.graph_relation) parts.push(provenance.graph_relation);
  if (typeof provenance.rerank_score === "number") {
    parts.push(`rerank ${provenance.rerank_score.toFixed(2)}`);
  } else if (typeof provenance.fused_score === "number") {
    parts.push(`fused ${provenance.fused_score.toFixed(3)}`);
  }
  return parts.join(" \u00b7 ");
}

function RetrievedChunk({ chunk }) {
  const [open, setOpen] = useState(false);
  const location = `${chunk.path}:${chunk.start_line}-${chunk.end_line}`;
  return html`
    <div class="chunk">
      <button onClick=${() => setOpen(!open)}>
        <span>${open ? "\u25be" : "\u25b8"}</span>
        <span>${chunk.symbol || location}</span>
        <span class="src">${provenanceLabel(chunk.provenance)}</span>
      </button>
      ${open ? html`<pre>${chunk.content}</pre>` : null}
    </div>
  `;
}

export function CodePanel({ inspected, retrievedFiles, chunks, diff, patch }) {
  const stat = diffStat(diff);
  return html`
    <div class="stack">
      <${Panel}
        title="Generated patch"
        count=${diff
          ? `${stat.files} file(s) +${stat.added} \u2212${stat.removed}`
          : null}
        flush=${true}
      >
        ${patch
          ? html`<div class="body" style=${{ paddingBottom: 0 }}>
              <div class="dim" style=${{ fontSize: "12.5px" }}>${patch.summary}</div>
            </div>`
          : null}
        <${Diff} text=${diff} />
      <//>

      <${Panel} title="Files inspected" count=${(inspected || []).length} flush=${true}>
        ${(inspected || []).length === 0
          ? html`<${Empty}>Nothing read yet.<//>`
          : html`<div class="filelist">
              ${inspected.map(
                (path) => html`<div key=${path} class="f">
                  <span>${path}</span>
                  ${(retrievedFiles || []).includes(path)
                    ? html`<span class="tag">retrieved</span>`
                    : null}
                </div>`
              )}
            </div>`}
      <//>

      <${Panel} title="Retrieved context" count=${(chunks || []).length} flush=${true}>
        ${(chunks || []).length === 0
          ? html`<${Empty}>No chunks retrieved yet.<//>`
          : html`<div>
              ${chunks.map(
                (chunk, index) =>
                  html`<${RetrievedChunk} key=${`${chunk.path}:${chunk.start_line}:${index}`} chunk=${chunk} />`
              )}
            </div>`}
      <//>
    </div>
  `;
}

export function ChecksPanel({ targeted, full, security, lint }) {
  const rows = [];
  if (targeted) {
    rows.push({
      label: "Targeted tests",
      ok: targeted.failed === 0 && targeted.errors === 0,
      value: `${targeted.passed} passed, ${targeted.failed + targeted.errors} failed`,
    });
  }
  if (full) {
    rows.push({
      label: "Full suite",
      ok: full.failed === 0 && full.errors === 0,
      value: `${full.passed}/${
        full.passed + full.failed + full.errors + full.skipped
      } passed`,
    });
  }
  if (lint) {
    rows.push({
      label: `Lint (${lint.tool})`,
      ok: lint.ok,
      value: lint.ok ? "passed" : `${lint.issue_count} issue(s)`,
    });
  }
  if (security) {
    rows.push({
      label: `Security (${security.backend})`,
      ok: security.ok,
      value: security.ok
        ? `${security.files_scanned} files, clean`
        : `${security.findings.length} finding(s)`,
    });
  }

  return html`
    <${Panel} title="Checks" flush=${true}>
      ${rows.length === 0
        ? html`<${Empty}>No checks have run yet.<//>`
        : html`<div class="gates">
            ${rows.map(
              (row) => html`<div key=${row.label} class="gate">
                <${Badge} tone=${row.ok ? "ok" : "fail"}>${row.ok ? "PASS" : "FAIL"}<//>
                <span class="label">${row.label}</span>
                <span class="value">${row.value}</span>
              </div>`
            )}
          </div>`}
      ${full && full.failures && full.failures.length
        ? html`<div class="body">
            <div class="faint" style=${{ marginBottom: "5px", fontSize: "11px" }}>
              Failing tests
            </div>
            ${full.failures.slice(0, 8).map(
              (failure) => html`<div
                key=${failure.node_id}
                class="mono"
                style=${{ fontSize: "11px", color: "#ffa198" }}
              >
                ${failure.node_id}
              </div>`
            )}
          </div>`
        : null}
      ${security && security.findings && security.findings.length
        ? html`<div class="body">
            ${security.findings.slice(0, 8).map(
              (finding, index) => html`<div
                key=${index}
                class="mono"
                style=${{ fontSize: "11px", color: "#f0c674" }}
              >
                ${finding.rule_id} ${finding.path}:${finding.line}
              </div>`
            )}
          </div>`
        : null}
    <//>
  `;
}
