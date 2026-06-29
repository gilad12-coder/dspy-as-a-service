"use client";

import { Globe, Wand2 } from "lucide-react";
import { msg } from "@/shared/lib/messages";
import { type Locale, LOCALES, LOCALE_REGISTRY } from "@/shared/lib/locale";
import { useLocale } from "@/shared/providers/locale-provider";
import { Button } from "@/shared/ui/primitives/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/ui/primitives/select";
import { SettingsRow } from "@/shared/ui/settings-row";

/**
 * Surface the interface-language controls that otherwise live only in the header
 * LanguageSwitcher popover. Both rows drive the existing LocaleProvider context —
 * no new state.
 *
 * Returns:
 *   The language settings tab body.
 */
export function LanguageTab() {
  const { locale, setLocale, isAuto, resetToAuto } = useLocale();

  return (
    <div className="space-y-1">
      <SettingsRow
        icon={Globe}
        label={msg("settings.language.interface.label")}
        description={msg("settings.language.interface.description")}
      >
        <Select value={locale} onValueChange={(next) => setLocale(next as Locale)}>
          <SelectTrigger className="min-w-[160px]">
            <SelectValue>{LOCALE_REGISTRY[locale].nativeName}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {LOCALES.map((tag) => {
              const entry = LOCALE_REGISTRY[tag];
              return (
                <SelectItem key={tag} value={tag}>
                  <span className="flex flex-col gap-0.5">
                    <span dir={entry.dir}>{entry.nativeName}</span>
                    <span className="text-xs text-muted-foreground">{entry.englishName}</span>
                  </span>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      </SettingsRow>

      <SettingsRow
        icon={Wand2}
        label={msg("settings.language.auto.label")}
        description={msg("settings.language.auto.description")}
      >
        <Button variant="outline" size="sm" onClick={resetToAuto} disabled={isAuto}>
          {msg("settings.language.auto.action")}
        </Button>
      </SettingsRow>
    </div>
  );
}
