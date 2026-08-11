"use client";

import * as React from "react";
import { signOut, useSession } from "next-auth/react";
import { toast } from "react-toastify";
import { At, ChartBar, CircleNotch, Database, DownloadSimple, Trash, User } from "@/shared/ui/icons";

import { msg, formatMsg } from "@/shared/lib/messages";
import { tI18n } from "@/shared/lib/i18n";
import {
  ApiError,
  deleteAccount,
  exportAccountData,
  getSecurityStatus,
  invalidateCache,
} from "@/shared/lib/api";
import { isTelemetryOptedOut, setTelemetryOptOut } from "@/shared/lib/telemetry/client";
import { SettingsRow } from "@/shared/ui/settings-row";
import { CopyButton } from "@/shared/ui/copy-button";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { Label } from "@/shared/ui/primitives/label";
import { Switch } from "@/shared/ui/primitives/switch";
import { Separator } from "@/shared/ui/primitives/separator";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";

/** Localize a data-action failure: semantic backend codes when present. */
function describeError(err: unknown, fallback: string): string {
  if (err instanceof ApiError && err.code) return tI18n(err.code);
  return fallback;
}

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

/** Privacy & data settings: telemetry opt-out, cache clearing, data export, and account deletion. */
export function PrivacyTab() {
  const { data: session } = useSession();
  // Seed once on mount; the switch is the inverse — ON means analytics are sent.
  const [optedOut, setOptedOut] = React.useState(() => isTelemetryOptedOut());
  const [exporting, setExporting] = React.useState(false);
  const [deleteOpen, setDeleteOpen] = React.useState(false);
  const [password, setPassword] = React.useState("");
  const [confirmEmail, setConfirmEmail] = React.useState("");
  const [deleting, setDeleting] = React.useState(false);
  // null until the security probe resolves; OAuth-only identities have no password.
  const [hasPassword, setHasPassword] = React.useState<boolean | null>(null);

  const signedOut = !session?.user;
  const email = session?.user?.email ?? "";

  React.useEffect(() => {
    if (signedOut) return;
    getSecurityStatus()
      .then((s) => setHasPassword(s.has_password))
      .catch(() => setHasPassword(null));
  }, [signedOut]);

  const handleClearCache = React.useCallback(() => {
    invalidateCache();
    toast.success(msg("settings.privacy.clear_cache.success"));
  }, []);

  const handleExport = React.useCallback(async () => {
    setExporting(true);
    try {
      const bundle = await exportAccountData();
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "skynet-account-export.json";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      toast.success(msg("settings.privacy.export.success"));
    } catch (err) {
      toast.error(describeError(err, msg("settings.privacy.export.error")));
    } finally {
      setExporting(false);
    }
  }, []);

  function openDeleteDialog() {
    setPassword("");
    setConfirmEmail("");
    setDeleteOpen(true);
  }

  async function handleDelete(e: React.FormEvent) {
    e.preventDefault();
    setDeleting(true);
    try {
      await deleteAccount(password);
      // The account is gone; drop the local session and land on the sign-in page.
      setDeleteOpen(false);
      await signOut({ callbackUrl: "/" });
    } catch (err) {
      toast.error(describeError(err, msg("settings.privacy.delete.error")));
    } finally {
      setDeleting(false);
    }
  }

  // A "type your email to confirm" gate; the backend re-checks the password.
  const confirmMatches = confirmEmail.trim().toLowerCase() === email.toLowerCase();
  const canDelete = confirmMatches && (hasPassword === false || password.length > 0);

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
          value={email}
          signedOut={signedOut}
        />
      </div>

      <Separator />

      <div className="space-y-1">
        <SettingsRow
          icon={DownloadSimple}
          label={msg("settings.privacy.export.label")}
          description={msg("settings.privacy.export.description")}
        >
          <Button
            variant="outline"
            size="sm"
            disabled={signedOut || exporting}
            onClick={() => void handleExport()}
            className="gap-1.5"
          >
            {exporting && <CircleNotch className="size-3.5 animate-spin" aria-hidden="true" />}
            {msg("settings.privacy.export.action")}
          </Button>
        </SettingsRow>

        <SettingsRow
          icon={Trash}
          label={msg("settings.privacy.delete.label")}
          description={msg("settings.privacy.delete.description")}
        >
          <Button
            variant="destructive"
            size="sm"
            disabled={signedOut}
            onClick={openDeleteDialog}
          >
            {msg("settings.privacy.delete.action")}
          </Button>
        </SettingsRow>
      </div>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{msg("settings.privacy.delete.dialog.title")}</DialogTitle>
            <DialogDescription>{msg("settings.privacy.delete.dialog.hint")}</DialogDescription>
          </DialogHeader>
          <form onSubmit={handleDelete} className="space-y-3">
            {hasPassword !== false && (
              <div>
                <Label
                  htmlFor="delete-password"
                  className="mb-1.5 block text-xs font-medium text-muted-foreground"
                >
                  {msg("settings.privacy.delete.dialog.password_label")}
                </Label>
                <Input
                  id="delete-password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  dir="ltr"
                />
              </div>
            )}
            <div>
              <Label
                htmlFor="delete-confirm"
                className="mb-1.5 block text-xs font-medium text-muted-foreground"
              >
                {formatMsg("settings.privacy.delete.dialog.confirm_label", { email })}
              </Label>
              <Input
                id="delete-confirm"
                value={confirmEmail}
                onChange={(e) => setConfirmEmail(e.target.value)}
                autoComplete="off"
                autoFocus
                dir="ltr"
              />
            </div>
            <Button
              type="submit"
              variant="destructive"
              disabled={deleting || !canDelete}
              className="w-full gap-2"
            >
              {deleting && <CircleNotch className="size-4 animate-spin" aria-hidden="true" />}
              {msg("settings.privacy.delete.dialog.confirm")}
            </Button>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
