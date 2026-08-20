import { html, useState } from "../lib/html.js";

export function Panel({ title, count, children, flush }) {
  return html`
    <div class="panel">
      ${title &&
      html`<header>
        <span>${title}</span>
        ${count !== undefined && count !== null
          ? html`<span class="count">${count}</span>`
          : null}
      </header>`}
      <div class=${flush ? "body flush" : "body"}>${children}</div>
    </div>
  `;
}

export function Badge({ tone = "neutral", children }) {
  return html`<span class=${`badge ${tone}`}>${children}</span>`;
}

export function Chip({ tone = "", label, value }) {
  return html`<span class=${`chip ${tone}`}>
    <i class="dot"></i>${label}${value !== undefined
      ? html` <b class="mono">${value}</b>`
      : null}
  </span>`;
}

export function Metric({ label, value, small }) {
  return html`<div class="metric">
    <div class="k">${label}</div>
    <div class=${small ? "v small" : "v"}>${value}</div>
  </div>`;
}

export function Empty({ children }) {
  return html`<div class="empty">${children}</div>`;
}

export function Tabs({ tabs, active, onChange }) {
  return html`<div class="tabs">
    ${tabs.map(
      (tab) => html`<button
        key=${tab.id}
        class=${tab.id === active ? "active" : ""}
        onClick=${() => onChange(tab.id)}
      >
        ${tab.label}
        ${tab.count !== undefined && tab.count !== null
          ? html`<span class="tab-count">${tab.count}</span>`
          : null}
      </button>`
    )}
  </div>`;
}

export function Bar({ value, tone = "" }) {
  const width = Math.max(0, Math.min(1, value || 0)) * 100;
  return html`<div class=${`bar ${tone}`}><i style=${{ width: `${width}%` }}></i></div>`;
}

export function ErrorBox({ children }) {
  if (!children) return null;
  return html`<div class="error-box">${children}</div>`;
}

/** A collapsible block; collapsed by default so long output does not dominate. */
export function Collapse({ label, children, open: initiallyOpen = false }) {
  const [open, setOpen] = useState(initiallyOpen);
  return html`
    <div>
      <button class="ghost" onClick=${() => setOpen(!open)}>
        ${open ? "\u25be" : "\u25b8"} ${label}
      </button>
      ${open ? children : null}
    </div>
  `;
}
