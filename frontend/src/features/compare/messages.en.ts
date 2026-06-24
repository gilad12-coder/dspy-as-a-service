// English UI strings for the compare slice. Edit directly; keys missing here fall
// back to the Hebrew slice via msg(), so partial translations are safe.

import type { compareMessages } from "./messages";

export const compareMessagesEn: Partial<Record<keyof typeof compareMessages, string>> = {
  "compare.load_error": "Failed to load optimizations",
  "compare.mismatch": "You can only compare runs from the same test set and with the same metric",
  "compare.cap_reached": "You can compare up to 4 optimizations at a time",
  "compare.partial_load": "Some runs didn't load and were excluded from the comparison",
};
