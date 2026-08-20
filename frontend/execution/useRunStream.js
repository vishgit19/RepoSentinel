import { useState, useEffect, useRef, useCallback } from "../lib/html.js";
import { api, subscribe } from "../lib/api.js";

const EVENT_TYPES = [
  "run_created",
  "run_started",
  "status",
  "timeline",
  "step",
  "metrics",
  "state",
  "run_finished",
  "stream_end",
  "heartbeat",
];

const TERMINAL = new Set(["approved", "rejected", "succeeded", "failed", "aborted"]);

function emptyRun() {
  return {
    runId: null,
    status: "idle",
    statusDetail: "",
    steps: [],
    events: [],
    stepCounts: {},
    metrics: {},
    trace: null,
    data: {},
    info: null,
    streaming: false,
  };
}

/**
 * Subscribes to a run's SSE stream and folds its events into view state.
 *
 * The bus replays the whole timeline to a late subscriber, so reconnecting or
 * reopening a finished run produces the same view as having watched it live.
 * Timeline events are keyed by sequence number to stay idempotent across a
 * replay.
 */
export function useRunStream() {
  const [run, setRun] = useState(emptyRun);
  const closeRef = useRef(null);
  const seenSeq = useRef(new Set());

  const stop = useCallback(() => {
    if (closeRef.current) {
      closeRef.current();
      closeRef.current = null;
    }
  }, []);

  useEffect(() => stop, [stop]);

  const attach = useCallback(
    (runId, initial = {}) => {
      stop();
      seenSeq.current = new Set();
      setRun({ ...emptyRun(), runId, status: "queued", streaming: true, ...initial });

      closeRef.current = subscribe(
        `/api/runs/${runId}/events`,
        EVENT_TYPES,
        (event) => {
          setRun((previous) => reduce(previous, event, seenSeq.current));
        },
        () => setRun((previous) => ({ ...previous, streaming: false }))
      );
    },
    [stop]
  );

  /** Load a finished run from the store and render it as a completed timeline. */
  const load = useCallback(
    async (runId) => {
      stop();
      const record = await api.getRun(runId);
      const state = record.state || {};
      seenSeq.current = new Set((record.events || []).map((event) => event.seq));
      setRun({
        ...emptyRun(),
        runId,
        status: record.status,
        events: record.events || [],
        steps: stepsFromEvents(record.events || []),
        stepCounts: countNodes(record.events || []),
        metrics: state.metrics || {},
        trace: null,
        data: dataFromState(state),
        info: {
          model: { model: record.model, provider: record.provider },
          strategy: record.strategy,
          memory_enabled: Boolean(record.memory_enabled),
        },
        streaming: false,
      });
      if (record.live) attach(runId);
      return record;
    },
    [stop, attach]
  );

  const reset = useCallback(() => {
    stop();
    setRun(emptyRun());
  }, [stop]);

  const patch = useCallback((partial) => {
    setRun((previous) => ({ ...previous, ...partial }));
  }, []);

  return { run, attach, load, reset, stop, patch };
}

function reduce(state, event, seen) {
  switch (event.type) {
    case "run_created":
      return { ...state, steps: event.steps || state.steps };

    case "run_started":
      return { ...state, info: event, status: "running" };

    case "status":
      return {
        ...state,
        status: event.status || state.status,
        statusDetail: event.detail || "",
      };

    case "timeline": {
      const incoming = event.event;
      if (!incoming || seen.has(incoming.seq)) return state;
      seen.add(incoming.seq);
      const counts = { ...state.stepCounts };
      counts[incoming.node] = (counts[incoming.node] || 0) + 1;
      return { ...state, events: [...state.events, incoming], stepCounts: counts };
    }

    case "step":
      return {
        ...state,
        steps: state.steps.map((step) =>
          step.node === event.node ? { ...step, status: event.status } : step
        ),
      };

    case "metrics":
      return { ...state, metrics: { ...state.metrics, ...event.metrics } };

    case "state":
      return { ...state, data: { ...state.data, ...event.data } };

    case "run_finished":
      return {
        ...state,
        status: event.status || state.status,
        metrics: { ...state.metrics, ...(event.metrics || {}) },
        trace: event.trace || state.trace,
      };

    case "stream_end":
      return { ...state, streaming: false };

    default: // heartbeat
      return state;
  }
}

function stepsFromEvents(events) {
  // A reopened run has no `step` events (they are transient), so the rail is
  // reconstructed from the timeline: last status per node wins.
  const order = [];
  const byNode = new Map();
  events.forEach((event) => {
    if (!byNode.has(event.node)) order.push(event.node);
    byNode.set(event.node, event.status);
  });
  return order.map((node) => ({
    node,
    label: node.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()),
    status: byNode.get(node),
  }));
}

function countNodes(events) {
  const counts = {};
  events.forEach((event) => {
    counts[event.node] = (counts[event.node] || 0) + 1;
  });
  return counts;
}

function dataFromState(state) {
  const patches = state.patches || [];
  const latest = patches[patches.length - 1];
  const tests = state.test_results || [];
  const security = state.security_results || [];
  const lint = state.lint_results || [];
  return {
    diff: latest ? latest.diff : "",
    patch: latest
      ? {
          summary: latest.summary,
          files_changed: latest.files_changed,
          lines_added: latest.lines_added,
          lines_removed: latest.lines_removed,
        }
      : null,
    files_inspected: state.files_inspected || [],
    retrieved_files: (state.retrieved_context || []).map((chunk) => chunk.path),
    retrieved_chunks: state.retrieved_context || [],
    targeted_tests: tests.filter((report) => report.scope === "targeted").pop() || null,
    full_tests: tests.filter((report) => report.scope === "full").pop() || null,
    security: security[security.length - 1] || null,
    lint: lint[lint.length - 1] || null,
    final_report: state.final_report || null,
    safety_events: state.safety_events || [],
  };
}

export { TERMINAL };
