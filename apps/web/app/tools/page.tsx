"use client";

import { useEffect, useState } from "react";

import { AppChrome } from "../components/AppChrome";
import { request } from "../lib/api";
import type { ToolRow } from "../lib/types";

export default function ToolsPage() {
  const [tools, setTools] = useState<ToolRow[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    request<ToolRow[]>("/v1/tools/catalog")
      .then(setTools)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  return (
    <AppChrome status={`${tools.length} approved`}>
      <div className="page-head">
        <div>
          <p className="kicker">Registry</p>
          <h1>Tool catalog</h1>
          <p>
            Approved rows from <code>GET /v1/tools/catalog</code>. High-risk tools stay listed here; Demo Mode removes
            them from the model-visible catalog at run time.
          </p>
        </div>
      </div>
      {error && <p className="banner error">{error}</p>}
      <div className="table-card">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Risk</th>
              <th>Source</th>
              <th>Version</th>
              <th>Status</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {tools.map((tool) => (
              <tr key={tool.id}>
                <td className="mono">{tool.name}</td>
                <td>
                  <em className={`risk risk-${tool.risk_level}`}>{tool.risk_level}</em>
                </td>
                <td>{tool.source}</td>
                <td className="mono">{tool.version}</td>
                <td>{tool.status}</td>
                <td className="desc">{tool.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </AppChrome>
  );
}
