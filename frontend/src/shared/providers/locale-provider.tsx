"use client";

import * as React from "react";
import {
  LOCALE_COOKIE,
  LOCALE_COOKIE_MAX_AGE,
  isLocale,
  type Locale,
} from "@/shared/lib/locale";
import { setClientLocale } from "@/shared/lib/runtime-locale";

interface LocaleContextValue {
  locale: Locale;
  setLocale: (next: Locale) => void;
}

const LocaleContext = React.createContext<LocaleContextValue | null>(null);

/** Read the active locale and a setter from the nearest LocaleProvider. */
export function useLocale(): LocaleContextValue {
  const ctx = React.useContext(LocaleContext);
  if (!ctx) {
    throw new Error("useLocale must be used within a LocaleProvider");
  }
  return ctx;
}

/**
 * Provide the request-resolved locale to the client tree and a setter that
 * switches it.
 *
 * Switching writes the persistence cookie and does a full reload: the server is
 * the single source of truth for locale (it drives SSR text, `<html dir>`, and
 * metadata), so re-rendering from it guarantees a consistent result rather than
 * trying to flip thousands of already-rendered `msg()` outputs in place. A
 * language switch is deliberate and rare, so the reload cost is acceptable.
 *
 * Args:
 *   initialLocale: Locale resolved server-side for this request.
 *   children: App subtree.
 */
export function LocaleProvider({
  initialLocale,
  children,
}: {
  initialLocale: Locale;
  children: React.ReactNode;
}) {
  // Align the sync msg() module-global with the server-resolved locale before
  // any descendant renders, so the first client render matches SSR. A lazy
  // useState initializer runs exactly once per mount (server + client), ahead
  // of children, without the re-render churn of an effect.
  React.useState(() => {
    setClientLocale(initialLocale);
    return null;
  });

  const setLocale = React.useCallback(
    (next: Locale) => {
      if (!isLocale(next) || next === initialLocale) return;
      document.cookie = `${LOCALE_COOKIE}=${next};path=/;max-age=${LOCALE_COOKIE_MAX_AGE};samesite=lax`;
      window.location.reload();
    },
    [initialLocale],
  );

  const value = React.useMemo<LocaleContextValue>(
    () => ({ locale: initialLocale, setLocale }),
    [initialLocale, setLocale],
  );

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}
