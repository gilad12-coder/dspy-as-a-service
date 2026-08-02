"use client";

import * as React from "react";
import { useSession } from "next-auth/react";
import { toast } from "react-toastify";
import { At, ChartBar, Database, User } from "@/shared/ui/icons";

import { msg } from "@/shared/lib/messages";
import { invalidateCache } from "@/shared/lib/api";
import { isTelemetryOptedOut, setTelemetryOptOut } from "@/shared/lib/telemetry/client";
import { SettingsRow } from "@/shared/ui/settings-row";
import { CopyButton } from "@/shared/ui/copy-button";
import { Button } from "@/shared/ui/primitives/button";
import { Switch } from "@/shared/ui/primitives/switch";
import { Separator } from "@/shared/ui/primitives/separator";

/** A monospace value with the app-standard animated copy button. */
function CopyValueRow({
  icon,
  label,
  value,
  signedOut,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  signedOut: boolean;
}) {
  return (
    <SettingsRow icon={icon} label={label}>
      <span className="text-sm font-mono text-foreground" dir="ltr">
        {signedOut ? msg("settings.account.signed_out") : value}
      </span>
      <CopyButton text={value} ariaLabel={label} variant="outline" disabled={signedOut} />
    </SettingsRow>
  );
}

/** Privacy & data settings: telemetry opt-out, cache clearing, and account clipboard rows. */
export function PrivacyTab() {
  const { data: session } = useSession();
  // Seed once on mount; the switch is the inverse — ON means analytics are sent.
  const [optedOut, setOptedOut] = React.useState(() => isTelemetryOptedOut());

  const handleClearCache = React.useCallback(() => {
    invalidateCache();
    toast.success(msg("settings.privacy.clear_cache.success"));
  }, []);

  const signedOut = !session?.user;

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <SettingsRow
          icon={ChartBar}
          label={msg("settings.privacy.analytics.label")}
          description={msg("settings.privacy.analytics.description")}
        >
          <Switch
            checked={!optedOut}
            onCheckedChange={(on) => {
              setTelemetryOptOut(!on);
              setOptedOut(!on);
            }}
          />
        </SettingsRow>
      </div>

      <Separator />

      <div className="space-y-1">
        <SettingsRow
          icon={Database}
          label={msg("settings.privacy.clear_cache.label")}
          description={msg("settings.privacy.clear_cache.description")}
        >
          <Button variant="outline" size="sm" onClick={handleClearCache}>
            {msg("settings.privacy.clear_cache.action")}
          </Button>
        </SettingsRow>

        <CopyValueRow
          icon={User}
          label={msg("settings.privacy.copy_username.label")}
          value={session?.user?.name ?? ""}
          signedOut={signedOut}
        />

        <CopyValueRow
          icon={At}
          label={msg("settings.privacy.copy_email.label")}
          value={session?.user?.email ?? ""}
          signedOut={signedOut}
        />
      </div>
    </div>
  );
}
