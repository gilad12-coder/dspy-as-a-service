/**
 * Device class for the phone shell.
 *
 * Phones (viewport at most 767px — the same edge below which the sidebar goes
 * off-canvas) get a view-first shell: dashboard, run details, explore, plus the
 * few interactions that make sense on a phone (chat with a finished run, buying
 * credits, the agent panel). Authoring surfaces — new optimization, dataset
 * upload/edit, tagging, storage — are desktop-only and route to a notice.
 */
export type DeviceClass = "phone" | "desktop";

export const PHONE_MEDIA_QUERY = "(max-width: 767px)";

const DESKTOP_ONLY_PREFIXES = ["/submit", "/datasets", "/tagger", "/storage"] as const;

/** Whether `pathname` is an authoring surface the phone shell replaces with a notice. */
export function isDesktopOnlyPath(pathname: string): boolean {
  return DESKTOP_ONLY_PREFIXES.some((p) => pathname === p || pathname.startsWith(`${p}/`));
}

/**
 * First-paint device class from request headers.
 *
 * `Sec-CH-UA-Mobile` (`?1` = phone) is authoritative when present; otherwise a
 * `Mobile`/`Mobi` UA token minus iPad (tablets land at desktop widths anyway).
 * The client re-derives from `matchMedia` after mount, so this only has to be
 * right often enough to avoid a shell swap on the common phone paths.
 */
export function deviceClassFromRequest(
  userAgent: string | null,
  secChUaMobile: string | null,
): DeviceClass {
  if (secChUaMobile !== null) return secChUaMobile.trim() === "?1" ? "phone" : "desktop";
  if (!userAgent || /\biPad\b/i.test(userAgent)) return "desktop";
  return /\bMobi(le)?\b/i.test(userAgent) ? "phone" : "desktop";
}
