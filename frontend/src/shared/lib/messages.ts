/**
 * Aggregated Hebrew UI catalog for the application.
 *
 * Per-feature slices live in ``features/<name>/messages.ts`` (and
 * ``shared/messages/messages.ts`` for cross-feature copy) and are
 * hand-edited — there is no codegen step for UI strings. This file
 * just re-exports the union and provides ``msg`` / ``formatMsg``
 * helpers. ESLint blocks inline Hebrew literals outside the slice
 * files and ``i18n/locales/he.json``.
 *
 * Backend i18n codes (errors, validations) take a different path:
 * they live in ``i18n/locales/he.json``, are regenerated into
 * ``shared/lib/generated/i18n-catalog.ts`` by
 * ``scripts/generate_i18n.py``, and are resolved via ``tI18n``.
 */

import { formatTemplate } from "@/shared/lib/i18n";
import { getActiveLocale } from "@/shared/lib/runtime-locale";
import { submitMessages } from "@/features/submit/messages";
import { dashboardMessages } from "@/features/dashboard/messages";
import { sidebarMessages } from "@/features/sidebar/messages";
import { exploreMessages } from "@/features/explore/messages";
import { datasetsMessages } from "@/features/datasets/messages";
import { storageMessages } from "@/features/storage/messages";
import { compareMessages } from "@/features/compare/messages";
import { taggerMessages } from "@/features/tagger/messages";
import { tutorialMessages } from "@/features/tutorial/messages";
import { settingsMessages } from "@/features/settings/messages";
import { authMessages } from "@/features/auth/messages";
import { optimizationsMessages } from "@/features/optimizations/messages";
import { trajectoryMessages } from "@/features/trajectory/messages";
import { agentPanelMessages } from "@/features/agent-panel/messages";
import { sharedMessages } from "@/shared/messages/messages";
import { submitMessagesEn } from "@/features/submit/messages.en";
import { dashboardMessagesEn } from "@/features/dashboard/messages.en";
import { sidebarMessagesEn } from "@/features/sidebar/messages.en";
import { exploreMessagesEn } from "@/features/explore/messages.en";
import { datasetsMessagesEn } from "@/features/datasets/messages.en";
import { storageMessagesEn } from "@/features/storage/messages.en";
import { compareMessagesEn } from "@/features/compare/messages.en";
import { taggerMessagesEn } from "@/features/tagger/messages.en";
import { tutorialMessagesEn } from "@/features/tutorial/messages.en";
import { settingsMessagesEn } from "@/features/settings/messages.en";
import { authMessagesEn } from "@/features/auth/messages.en";
import { optimizationsMessagesEn } from "@/features/optimizations/messages.en";
import { trajectoryMessagesEn } from "@/features/trajectory/messages.en";
import { agentPanelMessagesEn } from "@/features/agent-panel/messages.en";
import { sharedMessagesEn } from "@/shared/messages/messages.en";

export const MESSAGES = {
  ...submitMessages,
  ...dashboardMessages,
  ...sidebarMessages,
  ...exploreMessages,
  ...datasetsMessages,
  ...storageMessages,
  ...compareMessages,
  ...taggerMessages,
  ...tutorialMessages,
  ...settingsMessages,
  ...authMessages,
  ...optimizationsMessages,
  ...trajectoryMessages,
  ...agentPanelMessages,
  ...sharedMessages,
} as const;

export type MessageKey = keyof typeof MESSAGES;
type MessageParams = Record<string, string | number>;

// English overlay. Each slice is Partial, so any key absent here resolves to
// its Hebrew template in msg() — partial translations render without holes.
const MESSAGES_EN: Partial<Record<MessageKey, string>> = {
  ...submitMessagesEn,
  ...dashboardMessagesEn,
  ...sidebarMessagesEn,
  ...exploreMessagesEn,
  ...datasetsMessagesEn,
  ...storageMessagesEn,
  ...compareMessagesEn,
  ...taggerMessagesEn,
  ...tutorialMessagesEn,
  ...settingsMessagesEn,
  ...authMessagesEn,
  ...optimizationsMessagesEn,
  ...trajectoryMessagesEn,
  ...agentPanelMessagesEn,
  ...sharedMessagesEn,
};

/**
 * Look up a user-facing string by key and optionally interpolate placeholders.
 *
 * Resolves against the active locale (`runtime-locale`): English uses the
 * overlay when the key is translated and falls back to Hebrew otherwise, so a
 * missing English string degrades to Hebrew rather than to the raw key.
 */
export function msg(key: MessageKey, params?: MessageParams): string {
  const locale = getActiveLocale();
  const template = (locale === "en" ? MESSAGES_EN[key] : undefined) ?? MESSAGES[key];
  return params ? formatTemplate(template, params, locale) : template;
}

export function formatMsg(key: MessageKey, params: MessageParams): string {
  return msg(key, params);
}
