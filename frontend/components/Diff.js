import { html } from "../lib/html.js";
import { Empty } from "./ui.js";

/**
 * Unified-diff renderer.
 *
 * Colouring a diff by line prefix is both cheaper and more accurate than
 * running a general syntax highlighter over patch text, and it is the
 * distinction a reviewer actually needs: what was added, what was removed.
 */
function lineClass(line) {
  if (line.startsWith("@@")) return "hunk";
  if (line.startsWith("+++") || line.startsWith("---")) return "meta";
  if (line.startsWith("diff ") || line.startsWith("index ")) return "meta";
  if (line.startsWith("+")) return "add";
  if (line.startsWith("-")) return "del";
  return "";
}

export function Diff({ text }) {
  if (!text || !text.trim()) {
    return html`<${Empty}>No diff yet. The agent has not modified any file.<//>`;
  }
  const lines = text.replace(/\n$/, "").split("\n");
  return html`<pre class="diff">${lines.map(
    (line, index) =>
      html`<span key=${index} class=${`ln ${lineClass(line)}`}>${line || " "}</span>`
  )}</pre>`;
}

export function diffStat(text) {
  if (!text) return { files: 0, added: 0, removed: 0 };
  let files = 0;
  let added = 0;
  let removed = 0;
  text.split("\n").forEach((line) => {
    if (line.startsWith("diff --git")) files += 1;
    else if (line.startsWith("+") && !line.startsWith("+++")) added += 1;
    else if (line.startsWith("-") && !line.startsWith("---")) removed += 1;
  });
  return { files, added, removed };
}
