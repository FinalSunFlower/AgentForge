"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { THEME_KEY } from "../lib/api";

const NAV = [
  { href: "/", label: "Playground" },
  { href: "/tools", label: "Tools" },
  { href: "/evals", label: "Evals" },
  { href: "/architecture", label: "Architecture" },
];

type AppChromeProps = {
  status?: string;
  apiLive?: boolean | null;
  right?: ReactNode;
  children: ReactNode;
  wide?: boolean;
};

export function AppChrome({ status, apiLive, right, children, wide }: AppChromeProps) {
  const pathname = usePathname();
  const [light, setLight] = useState(false);

  useEffect(() => {
    setLight(window.localStorage.getItem(THEME_KEY) === "light");
  }, []);

  useEffect(() => {
    window.localStorage.setItem(THEME_KEY, light ? "light" : "dark");
    document.documentElement.dataset.theme = light ? "light" : "dark";
  }, [light]);

  return (
    <div className={`app ${light ? "light" : ""}`}>
      <header className="topbar">
        <Link href="/" className="brand">
          <span className="brand-mark">AF</span>
          <span className="brand-name">AgentForge</span>
        </Link>
        <nav aria-label="Primary">
          {NAV.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link className={`nav-link ${active ? "active" : ""}`} href={item.href} key={item.href}>
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="top-actions">
          {apiLive !== undefined && (
            <span className={`api-pill ${apiLive ? "ok" : apiLive === false ? "down" : ""}`}>
              <span className="dot" />
              {apiLive ? "API" : apiLive === false ? "API down" : "API"}
            </span>
          )}
          {status && <span className="status-label">{status}</span>}
          <button type="button" className="ghost-btn" onClick={() => setLight((current) => !current)}>
            {light ? "Dark" : "Light"}
          </button>
          {right}
        </div>
      </header>
      <div className={`workspace ${wide ? "wide" : ""}`}>
        <main className="main">{children}</main>
      </div>
    </div>
  );
}
