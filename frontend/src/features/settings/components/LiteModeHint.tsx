"use client";

import * as React from "react";
import { Feather, X } from "lucide-react";
import { Button } from "@/shared/ui/primitives/button";
import { msg } from "@/shared/lib/messages";
import { useUserPrefs } from "../hooks/use-user-prefs";

const DISMISS_KEY = "skynet.lite-hint.dismissed";

/** Coarse "is this a constrained machine?" check. Save-Data is an explicit user
 * signal; deviceMemory/hardwareConcurrency are rough and often undefined, so a
 * miss just means no nudge (the Settings switch is always available). */
function looksLowResource(): boolean {
  if (typeof navigator === "undefined") return false;
  const nav = navigator as Navigator & {
    deviceMemory?: number;
    connection?: { saveData?: boolean };
  };
  if (nav.connection?.saveData === true) return true;
  if (typeof nav.deviceMemory === "number" && nav.deviceMemory <= 4) return true;
  if (typeof nav.hardwareConcurrency === "number" && nav.hardwareConcurrency <= 4) return true;
  return false;
}

/**
 * One-time, dismissible nudge toward Lite mode for machines that look
 * resource-constrained. Suggests, never forces; a dismissal is remembered in
 * localStorage so it never reappears. Renders nothing once Lite is on, once
 * dismissed, or when the hardware looks capable.
 */
export function LiteModeHint() {
  const { prefs, setPref } = useUserPrefs();
  const [show, setShow] = React.useState(false);

  React.useEffect(() => {
    if (prefs.liteMode) return;
    let dismissed = false;
    try {
      dismissed = window.localStorage.getItem(DISMISS_KEY) === "1";
    } catch {
      /* noop */
    }
    if (!dismissed && looksLowResource()) setShow(true);
  }, [prefs.liteMode]);

  const dismiss = React.useCallback(() => {
    setShow(false);
    try {
      window.localStorage.setItem(DISMISS_KEY, "1");
    } catch {
      /* noop */
    }
  }, []);

  const enable = React.useCallback(() => {
    setPref("liteMode", true);
    dismiss();
  }, [setPref, dismiss]);

  if (!show || prefs.liteMode) return null;

  return (
    <div
      role="status"
      className="fixed inset-x-0 bottom-4 z-40 mx-auto flex w-max max-w-[calc(100vw-2rem)] items-center gap-3 rounded-full border border-border bg-card px-4 py-2 shadow-md"
    >
      <Feather className="size-4 shrink-0 text-[#B04030]" aria-hidden="true" />
      <span className="text-sm text-foreground">{msg("app.shell.lite.hint.text")}</span>
      <Button size="sm" onClick={enable}>
        {msg("app.shell.lite.hint.action")}
      </Button>
      <button
        type="button"
        onClick={dismiss}
        aria-label={msg("app.shell.lite.hint.dismiss_aria")}
        className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
      >
        <X className="size-4" aria-hidden="true" />
      </button>
    </div>
  );
}
