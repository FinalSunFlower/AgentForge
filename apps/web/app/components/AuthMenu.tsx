"use client";

import { useState } from "react";

import type { User } from "../lib/types";

type AuthMenuProps = {
  user: User | null;
  busy: boolean;
  onLogin: (email: string, password: string, mode: "login" | "register") => Promise<void>;
  onLogout: () => void;
};

export function AuthMenu({ user, busy, onLogin, onLogout }: AuthMenuProps) {
  const [open, setOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  if (user) {
    return (
      <div className="auth-slot">
        <span className="user-chip" title={user.email}>
          {user.display_name || user.email}
        </span>
        <button type="button" className="ghost-btn" onClick={onLogout}>
          Sign out
        </button>
      </div>
    );
  }

  return (
    <div className="auth-slot">
      <button type="button" className="primary-btn compact" onClick={() => setOpen((current) => !current)}>
        Sign in
      </button>
      {open && (
        <form
          className="auth-pop"
          onSubmit={(event) => {
            event.preventDefault();
            void onLogin(email, password, "login").then(() => setOpen(false));
          }}
        >
          <label>
            Email
            <input
              autoFocus
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
            />
          </label>
          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="At least 12 characters"
            />
          </label>
          <div className="auth-actions">
            <button type="submit" className="primary-btn" disabled={busy || !email || password.length < 12}>
              Sign in
            </button>
            <button
              type="button"
              className="ghost-btn"
              disabled={busy || !email || password.length < 12}
              onClick={() => void onLogin(email, password, "register").then(() => setOpen(false))}
            >
              Create account
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
