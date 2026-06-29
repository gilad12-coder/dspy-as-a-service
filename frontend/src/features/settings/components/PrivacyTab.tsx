"use client";

import * as React from "react";
import { useSession } from "next-auth/react";
import { toast } from "react-toastify";
import { AtSign, ChartNoAxesColumn, Check, Copy, Database, User } from "lucide-react";

import { msg } from "@/shared/lib/messages";
import { invalidateCache } from "@/shared/lib/api";
import { isTelemetryOptedOut, setTelemetryOptOut } from "@/shared/lib/telemetry/client";
import { SettingsRow } from "@/shared/ui/settings-row";
import { Button } from "@/shared/ui/primitives/button";
import { Switch } from "@/shared/ui/primitives/switch";
import { Separator } from "@/shared/ui/primitives/separator";

/** A monospace value with an icon Button that swaps Copy→Check for ~1.5s after copying. */
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
  const [copied, setCopied] = React.useState(false);

  const handleCopy = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
    } catch {
      // Clipboard access can be blocked; the value stays visible to copy by hand.
    }
  }, [value]);

  React.useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(t);
  }, [copied]);

  return (
    <SettingsRow icon={icon} label={label}>
      <span className="text-sm font-mono text-foreground" dir="ltr">
        {signedOut ? msg("settings.account.signed_out") : value}
      </span>
      <Button
        variant="outline"
        size="icon-sm"
        onClick={handleCopy}
        disabled={signedOut}
        aria-label={label}
      >
        {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      </Button>
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
          icon={ChartNoAxesColumn}
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
          icon={AtSign}
          label={msg("settings.privacy.copy_email.label")}
          value={session?.user?.email ?? ""}
          signedOut={signedOut}
        />
      </div>
    </div>
  );
}
