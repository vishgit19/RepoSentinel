// Thin wrapper over the JSON API. Errors carry the server's `detail` message so
// the UI can show why something was refused rather than a bare status code.

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (exception) {
    if (exception && exception.name === "AbortError") {
      throw new Error("The request timed out. Try again.");
    }
    throw exception;
  }
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    const detail =
      (payload && (payload.detail || payload.message)) || `HTTP ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

export const api = {
  health: () => request("/api/health"),
  benchmarks: () => request("/api/benchmarks"),
  models: () => request("/api/models"),
  strategies: () => request("/api/strategies"),
  topology: () => request("/api/topology"),
  tools: () => request("/api/tools"),

  startRun: (body) =>
    request("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  listRuns: (limit = 50) => request(`/api/runs?limit=${limit}`),
  getRun: (runId) => request(`/api/runs/${runId}`),
  decide: (runId, approved, note = "") => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    return request(`/api/runs/${runId}/approval`, {
      method: "POST",
      body: JSON.stringify({ approved, note }),
      signal: controller.signal,
    }).finally(() => clearTimeout(timer));
  },
  deleteRun: (runId) => request(`/api/runs/${runId}`, { method: "DELETE" }),

  startEvaluation: (body) =>
    request("/api/evaluations", { method: "POST", body: JSON.stringify(body) }),
  listEvaluations: () => request("/api/evaluations"),
  getEvaluation: (suiteId) => request(`/api/evaluations/${suiteId}`),
};

/**
 * Subscribe to a Server-Sent Events endpoint.
 *
 * The server names each event after its `type`, and the browser's EventSource
 * only delivers named events to matching listeners, so every type we care
 * about is registered explicitly.
 */
export function subscribe(url, types, onEvent, onError) {
  const source = new EventSource(url);
  const handler = (message) => {
    try {
      onEvent(JSON.parse(message.data));
    } catch (error) {
      // A malformed frame should not tear down the whole stream.
      console.warn("unparsable event", error);
    }
  };
  types.forEach((type) => source.addEventListener(type, handler));
  source.onerror = () => {
    // EventSource retries on its own; only a closed socket is terminal.
    if (source.readyState === EventSource.CLOSED && onError) onError();
  };
  return () => source.close();
}
