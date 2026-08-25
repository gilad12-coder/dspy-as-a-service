const LOCALE_RESET_PREFIX = "skynet.tagger.interview.locale-reset:";

function resetKey(sessionId: string): string {
  return `${LOCALE_RESET_PREFIX}${sessionId}`;
}

/** Remember that a tagging interview must be restarted after a locale reload. */
export function markTaggerInterviewForLocaleReset(sessionId: string): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(resetKey(sessionId), "1");
  } catch {
    // A storage failure should not block the locale switch itself.
  }
}

/** Check whether a tagging-interview reset is waiting for one session. */
export function hasTaggerInterviewLocaleReset(sessionId: string): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.sessionStorage.getItem(resetKey(sessionId)) === "1";
  } catch {
    return false;
  }
}

/** Clear a completed tagging-interview reset request. */
export function clearTaggerInterviewLocaleReset(sessionId: string): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(resetKey(sessionId));
  } catch {
    // A storage failure should not block the interview from starting.
  }
}
