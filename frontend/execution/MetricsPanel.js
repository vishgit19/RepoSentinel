import { html } from "../lib/html.js";
import { Panel, Metric, Empty, Badge } from "../components/ui.js";
import { integer, money, seconds } from "../lib/format.js";

/**
 * The evaluation panel for a single run: did it work, and what did it cost.
 * Verdicts come from recorded tool output, never from the model's own claim.
 */
export function MetricsPanel({ metrics, trace, checks, verified, safety }) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return html`<${Panel} title="Run metrics">
      <${Empty}>Metrics appear once the run reaches its first checkpoint.<//>
    <//>`;
  }

  const { targeted, full, security, lint } = checks || {};
  const testsLabel = full
    ? `${full.passed}/${full.passed + full.failed + full.errors + full.skipped}`
    : targeted
    ? `${targeted.passed} targeted`
    : "\u2014";

  return html`
    <div class="stack">
      <${Panel} title="Outcome" flush=${true}>
        <div class="gates">
          <div class="gate">
            <${Badge} tone=${verified ? "ok" : "fail"}>
              ${verified ? "PASS" : "FAIL"}
            <//>
            <span class="label">Repair verified</span>
          </div>
          <div class="gate">
            <${Badge} tone=${full && full.failed + full.errors === 0 ? "ok" : "neutral"}>
              ${full ? (full.failed + full.errors === 0 ? "PASS" : "FAIL") : "\u2014"}
            <//>
            <span class="label">Full test suite</span>
            <span class="value">${testsLabel}</span>
          </div>
          <div class="gate">
            <${Badge} tone=${security ? (security.ok ? "ok" : "fail") : "neutral"}>
              ${security ? (security.ok ? "PASS" : "FAIL") : "\u2014"}
            <//>
            <span class="label">Security checks</span>
          </div>
          <div class="gate">
            <${Badge} tone=${lint ? (lint.ok ? "ok" : "warn") : "neutral"}>
              ${lint ? (lint.ok ? "PASS" : "WARN") : "\u2014"}
            <//>
            <span class="label">Lint</span>
          </div>
        </div>
      <//>

      <${Panel} title="Effort" flush=${true}>
        <div class="metrics">
          <${Metric} label="Tool calls" value=${integer(metrics.tool_calls)} />
          <${Metric} label="LLM calls" value=${integer(metrics.llm_calls)} />
          <${Metric} label="Retrieved chunks" value=${integer(metrics.retrieved_chunks)} />
          <${Metric} label="Retries" value=${integer(metrics.retries)} />
          <${Metric} label="Latency" value=${seconds(metrics.latency_ms)} />
          <${Metric} label="Est. cost" value=${money(metrics.cost_usd)} />
          <${Metric}
            label="Tokens"
            value=${integer(metrics.total_tokens)}
            small=${true}
          />
          <${Metric}
            label="Blocked calls"
            value=${integer(metrics.blocked_tool_calls)}
            small=${true}
          />
        </div>
      <//>

      ${safety && safety.length
        ? html`<${Panel} title="Guardrail events" count=${safety.length} flush=${true}>
            <div class="filelist">
              ${safety.map(
                (event, index) => html`<div key=${index} class="f">
                  <span>${event.kind}</span>
                  <span class="tag">${event.source || ""}</span>
                </div>`
              )}
            </div>
          <//>`
        : null}

      ${trace
        ? html`<${Panel} title="Trace" count=${`${trace.spans || 0} spans`} flush=${true}>
            <div class="metrics">
              ${Object.entries(trace.by_kind || {}).map(
                ([kind, count]) =>
                  html`<${Metric} key=${kind} label=${kind} value=${integer(count)} small=${true} />`
              )}
              <${Metric}
                label="span errors"
                value=${integer(trace.errors)}
                small=${true}
              />
              <${Metric}
                label="duration"
                value=${seconds(trace.duration_ms)}
                small=${true}
              />
            </div>
          <//>`
        : null}
    </div>
  `;
}
