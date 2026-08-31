"use client";

import { useState } from "react";

import { eventLabel, eventSummary, visibleTrace } from "../lib/events";
import type { EventRecord } from "../lib/types";

export function TraceList({ events }: { events: EventRecord[] }) {
  const rows = visibleTrace(events);
  const [open, setOpen] = useState<number | null>(null);
  if (rows.length === 0) {
    return (
      <div className="empty">
        <p>No durable events yet</p>
        <small>SSE frames land here. Token deltas stay in the transcript.</small>
      </div>
    );
  }
  return (
    <ol className="trace">
      {rows.map((event, index) => {
        const summary = eventSummary(event);
        const expanded = open === event.sequence;
        return (
          <li key={`${event.sequence}-${index}`}>
            <button type="button" className="trace-row" onClick={() => setOpen(expanded ? null : event.sequence)}>
              <span className="trace-seq">{String(event.sequence).padStart(2, "0")}</span>
              <span className={`trace-type type-${event.type.replace(/\./g, "-")}`}>{eventLabel(event.type)}</span>
              {summary && <span className="trace-sum">{summary}</span>}
            </button>
            {expanded && <pre className="trace-json">{JSON.stringify(event.payload, null, 2)}</pre>}
          </li>
        );
      })}
    </ol>
  );
}
