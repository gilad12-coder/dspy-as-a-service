/**
 * Tracks the session/run the user was most recently looking at, per kind, so the
 * sidebar's "Text tagging" and "Optimize" nav buttons can resume it on click —
 * but only within a short window after the user left it. Past that window the
 * button falls back to starting something new. State lives in localStorage so it
 * survives a reload and is read at click time (the window is measured live).
 */

type RecentKind = "tagger" | "optimization";

const STORAGE_KEY: Record<RecentKind, string> = {
  tagger: "skynet.recent.tagger",
  optimization: "skynet.recent.optimization",
};

/** How recently the user must have been in a session for the nav button to resume it. */
export const RESUME_WINDOW_MS = 60_000;

/** Record that the user is (or just was) viewing ``id`` of the given kind. */
export function markRecentSession(kind: RecentKind, id: string): void {
  if (typeof window === "undefined" || !id) return;
  try {
    window.localStorage.setItem(STORAGE_KEY[kind], JSON.stringify({ id, ts: Date.now() }));
  } catch {
    // localStorage unavailable (private mode / quota) — resume-on-return simply
    // degrades to always-new, which is harmless.
  }
}

/**
 * The id to resume for this kind, or null when there is none or the user left it
 * more than {@link RESUME_WINDOW_MS} ago (in which case the caller starts fresh).
 */
export function recentResumableId(kind: RecentKind): string | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY[kind]);
    if (!raw) return null;
    const { id, ts } = JSON.parse(raw) as { id?: unknown; ts?: unknown };
    if (typeof id === "string" && typeof ts === "number" && Date.now() - ts < RESUME_WINDOW_MS) {
      return id;
    }
  } catch {
    return null;
  }
  return null;
}
