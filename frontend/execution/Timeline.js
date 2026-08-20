import { html, useState, useEffect, useRef } from "../lib/html.js";
import { Panel, Empty } from "../components/ui.js";
import { STATUS_GLYPH, glyphClass, seconds, money, integer } from "../lib/format.js";

/**
 * The live execution timeline.
 *
 * This shows what the agent *did* - node, action, tool call, evidence, result,
 * timing and cost. It deliberately does not show raw model reasoning: the
 * backend only publishes summarised decisions written for a human reader.
 */
function TimelineEvent({ event }) {
  const [open, setOpen] = useState(false);
  const status = event.status || "running";
  const call = event.tool_call;
  const isTool = event.node === "tools" && Boolean(call);
  const expandable = Boolean(call || (event.detail || "").length > 240);

  return html`
    <div class="tl-event">
      <div class="tl-head" onClick=${() => expandable && setOpen(!open)}>
        <span class=${`tl-glyph ${glyphClass(status)} ${status === "running" ? "pulse" : ""}`}>
          ${STATUS_GLYPH[status] || STATUS_GLYPH.pending}
        </span>
        <span class="tl-node">${event.node}</span>
        <span class=${isTool ? "tl-title mono" : "tl-title"}>${event.title}</span>
        <span class="tl-time">
          ${event.duration_ms ? seconds(event.duration_ms) : ""}
          ${expandable ? (open ? " \u25be" : " \u25b8") : ""}
        </span>
      </div>

      ${event.detail
        ? html`<div class="tl-detail">
            ${open ? event.detail : truncate(event.detail, 240)}
          </div>`
        : null}

      ${(event.lines || []).length
        ? html`<div class="tl-lines">
            ${event.lines.map((line, index) => html`<div key=${index}>${line}</div>`)}
          </div>`
        : null}

      ${(event.evidence || []).length
        ? html`<div class="tl-evidence">
            ${event.evidence.map(
              (item, index) => html`<span key=${index} class="ev">${item}</span>`
            )}
          </div>`
        : null}

      ${open && call ? html`<${ToolDetail} call=${call} />` : null}
    </div>
  `;
}

function ToolDetail({ call }) {
  return html`
    <div class="tl-expand">
      <dl class="kv">
        <dt>tool</dt>
        <dd>${call.name}</dd>
        <dt>executed</dt>
        <dd>${call.executed ? "yes" : "no"}</dd>
        <dt>finding</dt>
        <dd>${call.ok ? "ok" : "not ok"}</dd>
        <dt>duration</dt>
        <dd>${seconds(call.duration_ms)}</dd>
        ${call.via_mcp ? html`<dt>via</dt><dd>MCP</dd>` : null}
        ${call.blocked
          ? html`<dt>blocked</dt><dd>${call.block_reason || "guardrail"}</dd>`
          : null}
      </dl>
      ${Object.keys(call.arguments || {}).length
        ? html`<pre class="out">${JSON.stringify(call.arguments, null, 2)}</pre>`
        : null}
      ${call.output ? html`<pre class="out">${call.output}</pre>` : null}
      ${call.error ? html`<pre class="out">${call.error}</pre>` : null}
    </div>
  `;
}

function truncate(text, limit) {
  const value = String(text);
  return value.length > limit ? `${value.slice(0, limit)}\u2026` : value;
}

export function Timeline({ events, live, tokens, cost }) {
  const endRef = useRef(null);
  const [pinned, setPinned] = useState(true);

  useEffect(() => {
    if (!pinned || !live || !endRef.current) return;
    const scroller = endRef.current.closest(".col-timeline-body");
    if (scroller) {
      scroller.scrollTop = scroller.scrollHeight;
      return;
    }
    endRef.current.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [events.length, pinned, live]);

  const header = live
    ? `${events.length} events \u00b7 ${integer(tokens)} tokens \u00b7 ${money(cost)}`
    : `${events.length} events`;

  return html`
    <${Panel} title="Agent execution" count=${header} flush=${true}>
      ${events.length === 0
        ? html`<${Empty}>
            Pick a problem and start an investigation. Every node, tool call,
            retrieval and test run appears here as it happens.
          <//>`
        : html`<div class="timeline">
            ${events.map(
              (event) => html`<${TimelineEvent} key=${event.event_id || event.seq} event=${event} />`
            )}
            <div ref=${endRef}></div>
          </div>`}
    <//>
    ${live && events.length > 6
      ? html`<div class="row">
          <label class="checkline">
            <input
              type="checkbox"
              checked=${pinned}
              onChange=${(e) => setPinned(e.target.checked)}
            />
            Follow live output
          </label>
        </div>`
      : null}
  `;
}
