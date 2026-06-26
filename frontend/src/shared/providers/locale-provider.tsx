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
  /** True when no locale cookie is set, i.e. the active locale was auto-detected. */
  isAuto: boolean;
  /** Clear the persisted choice and reload, reverting to Accept-Language detection. */
  resetToAuto: () => void;
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
  // Align the sync msg() locale module-global with the server-resolved locale
  // before any descendant renders, so the first client render matches SSR. A
  // lazy useState initializer runs exactly once per mount (server + client),
  // ahead of children, without the re-render churn of an effect. The catalog
  // itself is not seeded here: the browser reads it lazily from the
  // `window.__SKYNET_MESSAGES__` shim, and SSR of client components from the
  // request global the layout publishes (see runtime-messages.ts).
  React.useState(() => {
    setClientLocale(initialLocale);
    return null;
  });

  // Whether the active locale came from auto-detection (no cookie) vs an explicit
  // pick. Resolved post-mount so SSR (no document) and first client render agree;
  // any switch reloads the page, so this mount read is always current.
  const [isAuto, setIsAuto] = React.useState(false);
  React.useEffect(() => {
    const hasCookie = document.cookie
      .split("; ")
      .some((entry) => entry.startsWith(`${LOCALE_COOKIE}=`));
    setIsAuto(!hasCookie);
  }, []);

  const setLocale = React.useCallback(
    (next: Locale) => {
      // Re-picking the already-active explicit locale is a no-op; but in auto mode
      // the same pick still pins the choice (writes the cookie), so don't skip it.
      if (!isLocale(next) || (!isAuto && next === initialLocale)) return;
      document.cookie = `${LOCALE_COOKIE}=${next};path=/;max-age=${LOCALE_COOKIE_MAX_AGE};samesite=lax`;
      window.location.reload();
    },
    [initialLocale, isAuto],
  );

  const resetToAuto = React.useCallback(() => {
    document.cookie = `${LOCALE_COOKIE}=;path=/;max-age=0;samesite=lax`;
    window.location.reload();
  }, []);

  const value = React.useMemo<LocaleContextValue>(
    () => ({ locale: initialLocale, setLocale, isAuto, resetToAuto }),
    [initialLocale, setLocale, isAuto, resetToAuto],
  );

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}
