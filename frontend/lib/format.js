export const STATUS_GLYPH = {
  success: "\u2713", // check
  failure: "\u2717", // ballot x
  running: "\u2192", // arrow
  pending: "\u25cb", // open circle
  skipped: "\u2013", // en dash
  blocked: "!",
};

export function glyphClass(status) {
  return `g-${status || "pending"}`;
}

export function seconds(ms) {
  if (!ms && ms !== 0) return "\u2014";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function money(value) {
  if (value === null || value === undefined) return "\u2014";
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

export function integer(value) {
  if (value === null || value === undefined) return "\u2014";
  return Number(value).toLocaleString();
}

export function percent(value) {
  if (value === null || value === undefined) return "\u2014";
  return `${Math.round(value * 100)}%`;
}

export function ratio(value) {
  if (value === null || value === undefined) return "\u2014";
  return Number(value).toFixed(2);
}

export function timeAgo(epochSeconds) {
  if (!epochSeconds) return "";
  const delta = Date.now() / 1000 - epochSeconds;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

const STATUS_TONE = {
  approved: "ok",
  succeeded: "ok",
  rejected: "warn",
  awaiting_approval: "warn",
  running: "info",
  queued: "neutral",
  failed: "fail",
  aborted: "fail",
};

export function statusTone(status) {
  return STATUS_TONE[status] || "neutral";
}

/** Tool titles are already `name(args)`; everything else is prose. */
export function isToolTitle(node) {
  return node === "tools";
}
