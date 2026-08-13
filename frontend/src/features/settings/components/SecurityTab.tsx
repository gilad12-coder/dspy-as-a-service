"use client";

import * as React from "react";
import { toast } from "react-toastify";
import {
  Check,
  CircleNotch,
  DeviceMobile,
  Envelope,
  Fingerprint,
  PencilSimple,
  Plus,
  Trash,
  X,
} from "@/shared/ui/icons";
import { QRCodeSVG } from "qrcode.react";
import { browserSupportsWebAuthn, startRegistration } from "@simplewebauthn/browser";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { Label } from "@/shared/ui/primitives/label";
import { Switch } from "@/shared/ui/primitives/switch";
import { SettingsRow } from "@/shared/ui/settings-row";
import { CopyButton } from "@/shared/ui/copy-button";
import { msg, formatMsg } from "@/shared/lib/messages";
import { tI18n } from "@/shared/lib/i18n";
import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";
import {
  ApiError,
  deletePasskey,
  disableTotp,
  enableTotp,
  getPasskeyRegistrationOptions,
  getSecurityStatus,
  registerPasskey,
  renamePasskey,
  setEmailCodes,
  setupTotp,
  type SecurityStatus,
  type TotpSetup,
} from "@/shared/lib/api";

/** Localize a security API failure: semantic backend codes when present. */
function describeError(err: unknown): string {
  if (err instanceof ApiError && err.code) return tI18n(err.code);
  return msg("settings.security.error");
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(getActiveIntlLocale(), {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/**
 * Settings → Security: two-factor enrollment (authenticator app + emailed
 * codes) and passkey management. 2FA only guards password sign-ins, so the
 * toggles collapse to a provider-managed note for OAuth-only identities —
 * passkeys stay available to everyone.
 */
export function SecurityTab() {
  const [status, setStatus] = React.useState<SecurityStatus | null>(null);
  const [totpSetup, setTotpSetup] = React.useState<TotpSetup | null>(null);
  const [totpCode, setTotpCode] = React.useState("");
  const [recoveryCodes, setRecoveryCodes] = React.useState<string[] | null>(null);
  const [disableOpen, setDisableOpen] = React.useState(false);
  const [disableCode, setDisableCode] = React.useState("");
  const [passkeyOpen, setPasskeyOpen] = React.useState(false);
  const [passkeyName, setPasskeyName] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [deleting, setDeleting] = React.useState<string | null>(null);
  const [editingPasskey, setEditingPasskey] = React.useState<string | null>(null);
  const [editingPasskeyName, setEditingPasskeyName] = React.useState("");
  const [renaming, setRenaming] = React.useState<string | null>(null);
  const passkeysSupported = React.useMemo(() => browserSupportsWebAuthn(), []);

  const refresh = React.useCallback(() => {
    getSecurityStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);
  React.useEffect(refresh, [refresh]);

  async function beginTotpSetup() {
    setBusy(true);
    try {
      const setup = await setupTotp();
      setTotpCode("");
      setRecoveryCodes(null);
      setTotpSetup(setup);
    } catch (err) {
      toast.error(describeError(err));
    } finally {
      setBusy(false);
    }
  }

  async function confirmTotp(e: React.FormEvent) {
    e.preventDefault();
    if (!totpCode.trim()) return;
    setBusy(true);
    try {
      const result = await enableTotp(totpCode.trim());
      setRecoveryCodes(result.recovery_codes);
      toast.success(msg("settings.security.totp_enabled_toast"));
      refresh();
    } catch (err) {
      toast.error(describeError(err));
    } finally {
      setBusy(false);
    }
  }

  async function confirmDisable(e: React.FormEvent) {
    e.preventDefault();
    if (!disableCode.trim()) return;
    setBusy(true);
    try {
      await disableTotp(disableCode.trim());
      toast.success(msg("settings.security.totp_disabled_toast"));
      setDisableOpen(false);
      setDisableCode("");
      refresh();
    } catch (err) {
      toast.error(describeError(err));
    } finally {
      setBusy(false);
    }
  }

  async function toggleEmailCodes(enabled: boolean) {
    // Optimistic flip; a failure (e.g. no SMTP relay) rolls back via refresh.
    setStatus((s) => (s ? { ...s, email_2fa_enabled: enabled } : s));
    try {
      await setEmailCodes(enabled);
      toast.success(
        msg(
          enabled
            ? "settings.security.email_enabled_toast"
            : "settings.security.email_disabled_toast",
        ),
      );
    } catch (err) {
      toast.error(describeError(err));
    } finally {
      refresh();
    }
  }

  async function addPasskey(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      const options = await getPasskeyRegistrationOptions();
      const credential = await startRegistration(
        options as unknown as Parameters<typeof startRegistration>[0],
      );
      await registerPasskey(credential, passkeyName.trim());
      toast.success(msg("settings.security.passkeys.added"));
      setPasskeyOpen(false);
      setPasskeyName("");
      refresh();
    } catch (err) {
      // Dismissing the platform prompt is a cancel, not an error.
      if ((err as Error)?.name !== "NotAllowedError") toast.error(describeError(err));
    } finally {
      setBusy(false);
    }
  }

  async function removePasskey(credentialId: string) {
    setDeleting(credentialId);
    try {
      await deletePasskey(credentialId);
      toast.success(msg("settings.security.passkeys.deleted"));
      refresh();
    } catch (err) {
      toast.error(describeError(err));
    } finally {
      setDeleting(null);
    }
  }

  function beginPasskeyRename(passkey: SecurityStatus["passkeys"][number]) {
    setEditingPasskey(passkey.credential_id);
    setEditingPasskeyName(passkey.nickname);
  }

  function cancelPasskeyRename() {
    setEditingPasskey(null);
    setEditingPasskeyName("");
  }

  async function savePasskeyRename(credentialId: string) {
    const nickname = editingPasskeyName.trim();
    if (!nickname) return;
    setRenaming(credentialId);
    try {
      await renamePasskey(credentialId, nickname);
      toast.success(msg("settings.security.passkeys.renamed"));
      if (editingPasskey === credentialId) cancelPasskeyRename();
      refresh();
    } catch (err) {
      toast.error(describeError(err));
    } finally {
      setRenaming(null);
    }
  }

  function closeTotpDialog() {
    setTotpSetup(null);
    setTotpCode("");
    setRecoveryCodes(null);
    refresh();
  }

  if (!status) {
    return (
      <div className="flex items-center justify-center py-10 text-muted-foreground">
        <CircleNotch className="size-4 animate-spin" aria-hidden="true" />
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {status.has_password ? (
        <>
          <SettingsRow
            icon={DeviceMobile}
            label={msg("settings.security.totp.label")}
            description={msg("settings.security.totp.description")}
          >
            {status.totp_enabled ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setDisableCode("");
                  setDisableOpen(true);
                }}
              >
                {msg("settings.security.disable")}
              </Button>
            ) : (
              <Button size="sm" disabled={busy} onClick={() => void beginTotpSetup()}>
                {msg("settings.security.enable")}
              </Button>
            )}
          </SettingsRow>

          <SettingsRow
            icon={Envelope}
            label={msg("settings.security.email.label")}
            description={
              status.email_2fa_available
                ? msg("settings.security.email.description")
                : msg("settings.security.email.unavailable")
            }
          >
            <Switch
              checked={status.email_2fa_enabled}
              disabled={!status.email_2fa_available && !status.email_2fa_enabled}
              onCheckedChange={(v) => void toggleEmailCodes(v)}
            />
          </SettingsRow>
        </>
      ) : (
        <SettingsRow
          icon={DeviceMobile}
          label={msg("settings.security.provider_managed.label")}
          description={msg("settings.security.provider_managed.description")}
        >
          <span />
        </SettingsRow>
      )}

      <SettingsRow
        icon={Fingerprint}
        label={msg("settings.security.passkeys.label")}
        description={
          passkeysSupported
            ? msg("settings.security.passkeys.description")
            : msg("settings.security.passkeys.unsupported")
        }
      >
        <Button
          size="sm"
          disabled={!passkeysSupported || busy}
          onClick={() => {
            setPasskeyName("");
            setPasskeyOpen(true);
          }}
          className="gap-1.5"
        >
          <Plus className="size-3.5" aria-hidden="true" />
          {msg("settings.security.passkeys.add")}
        </Button>
      </SettingsRow>

      {status.passkeys.length === 0 ? (
        <p className="px-7 py-2 text-xs text-muted-foreground/80">
          {msg("settings.security.passkeys.empty")}
        </p>
      ) : (
        <ul className="space-y-1 ps-7">
          {status.passkeys.map((passkey) => (
            <li
              key={passkey.credential_id}
              className="flex items-center justify-between gap-3 rounded-lg border border-border/40 px-3 py-2"
            >
              <div className="min-w-0 flex-1">
                {editingPasskey === passkey.credential_id ? (
                  <form
                    className="flex min-w-0 items-center gap-1.5"
                    onSubmit={(event) => {
                      event.preventDefault();
                      void savePasskeyRename(passkey.credential_id);
                    }}
                  >
                    <Input
                      value={editingPasskeyName}
                      onChange={(event) => setEditingPasskeyName(event.target.value)}
                      aria-label={msg("settings.security.passkeys.name_label")}
                      maxLength={64}
                      autoFocus
                      className="h-8 min-w-0 rounded-lg px-2 text-sm"
                    />
                    <Button
                      type="submit"
                      variant="ghost"
                      size="icon-sm"
                      aria-label={msg("settings.security.passkeys.rename_save")}
                      disabled={!editingPasskeyName.trim() || renaming === passkey.credential_id}
                    >
                      {renaming === passkey.credential_id ? (
                        <CircleNotch className="size-3.5 animate-spin" />
                      ) : (
                        <Check className="size-3.5" />
                      )}
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-label={msg("settings.security.passkeys.rename_cancel")}
                      disabled={renaming === passkey.credential_id}
                      onClick={cancelPasskeyRename}
                    >
                      <X className="size-3.5" />
                    </Button>
                  </form>
                ) : (
                  <p className="truncate text-sm font-medium text-foreground">{passkey.nickname}</p>
                )}
                <p className="text-xs text-muted-foreground">
                  {formatMsg("settings.security.passkeys.created", {
                    date: formatDate(passkey.created_at),
                  })}
                  {" · "}
                  {passkey.last_used_at
                    ? formatMsg("settings.security.passkeys.last_used", {
                        date: formatDate(passkey.last_used_at),
                      })
                    : msg("settings.security.passkeys.never_used")}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                {editingPasskey !== passkey.credential_id && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={msg("settings.security.passkeys.rename")}
                    disabled={renaming !== null}
                    onClick={() => beginPasskeyRename(passkey)}
                  >
                    <PencilSimple className="size-3.5" />
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label={msg("settings.security.passkeys.delete_aria")}
                  disabled={
                    deleting === passkey.credential_id || renaming === passkey.credential_id
                  }
                  onClick={() => void removePasskey(passkey.credential_id)}
                >
                  {deleting === passkey.credential_id ? (
                    <CircleNotch className="size-3.5 animate-spin" />
                  ) : (
                    <Trash className="size-3.5" />
                  )}
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}

      <Dialog open={totpSetup !== null} onOpenChange={(open) => !open && closeTotpDialog()}>
        <DialogContent data-settings-text-buttons className="sm:max-w-md">
          {recoveryCodes ? (
            <>
              <DialogHeader>
                <DialogTitle>{msg("settings.security.totp_dialog.recovery_title")}</DialogTitle>
                <DialogDescription>
                  {msg("settings.security.totp_dialog.recovery_hint")}
                </DialogDescription>
              </DialogHeader>
              <div
                className="grid grid-cols-2 gap-2 rounded-lg bg-accent/40 p-3 font-mono text-sm"
                dir="ltr"
              >
                {recoveryCodes.map((code) => (
                  <span key={code}>{code}</span>
                ))}
              </div>
              <div className="flex items-center justify-between">
                <CopyButton
                  text={recoveryCodes.join("\n")}
                  ariaLabel={msg("settings.security.copy_recovery_aria")}
                />
                <Button onClick={closeTotpDialog}>
                  {msg("settings.security.totp_dialog.done")}
                </Button>
              </div>
            </>
          ) : (
            totpSetup && (
              <>
                <DialogHeader>
                  <DialogTitle>{msg("settings.security.totp_dialog.title")}</DialogTitle>
                  <DialogDescription>{msg("settings.security.totp_dialog.scan")}</DialogDescription>
                </DialogHeader>
                <div className="flex flex-col items-center gap-3">
                  <div className="rounded-lg bg-white p-3">
                    <QRCodeSVG value={totpSetup.otpauth_url} size={168} />
                  </div>
                  <div
                    className="flex items-center gap-2 font-mono text-xs text-muted-foreground"
                    dir="ltr"
                  >
                    <span className="break-all">{totpSetup.secret}</span>
                    <CopyButton
                      text={totpSetup.secret}
                      ariaLabel={msg("settings.security.copy_secret_aria")}
                    />
                  </div>
                </div>
                <form onSubmit={confirmTotp} className="space-y-3">
                  <div>
                    <Label
                      htmlFor="totp-confirm-code"
                      className="mb-1.5 block text-xs font-medium text-muted-foreground"
                    >
                      {msg("settings.security.totp_dialog.code_label")}
                    </Label>
                    <Input
                      id="totp-confirm-code"
                      value={totpCode}
                      onChange={(e) => setTotpCode(e.target.value)}
                      placeholder="123456"
                      autoFocus
                      autoComplete="one-time-code"
                      inputMode="numeric"
                      dir="ltr"
                      className="text-left"
                    />
                  </div>
                  <Button
                    type="submit"
                    disabled={busy || !totpCode.trim()}
                    className="w-full gap-2"
                  >
                    {busy && <CircleNotch className="size-4 animate-spin" />}
                    {msg("settings.security.totp_dialog.verify")}
                  </Button>
                </form>
              </>
            )
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={disableOpen} onOpenChange={setDisableOpen}>
        <DialogContent data-settings-text-buttons className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{msg("settings.security.disable_dialog.title")}</DialogTitle>
            <DialogDescription>{msg("settings.security.disable_dialog.hint")}</DialogDescription>
          </DialogHeader>
          <form onSubmit={confirmDisable} className="space-y-3">
            <Input
              value={disableCode}
              onChange={(e) => setDisableCode(e.target.value)}
              placeholder="123456"
              autoFocus
              autoComplete="one-time-code"
              dir="ltr"
              className="text-left"
              aria-label={msg("settings.security.totp_dialog.code_label")}
            />
            <Button
              type="submit"
              variant="destructive"
              disabled={busy || !disableCode.trim()}
              className="w-full gap-2"
            >
              {busy && <CircleNotch className="size-4 animate-spin" />}
              {msg("settings.security.disable")}
            </Button>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog open={passkeyOpen} onOpenChange={setPasskeyOpen}>
        <DialogContent data-settings-text-buttons className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{msg("settings.security.passkeys.add")}</DialogTitle>
            <DialogDescription>{msg("settings.security.passkeys.description")}</DialogDescription>
          </DialogHeader>
          <form onSubmit={addPasskey} className="space-y-3">
            <div>
              <Label
                htmlFor="passkey-name"
                className="mb-1.5 block text-xs font-medium text-muted-foreground"
              >
                {msg("settings.security.passkeys.name_label")}
              </Label>
              <Input
                id="passkey-name"
                value={passkeyName}
                onChange={(e) => setPasskeyName(e.target.value)}
                placeholder={msg("settings.security.passkeys.name_placeholder")}
                autoFocus
                maxLength={64}
              />
            </div>
            <Button type="submit" disabled={busy} className="w-full gap-2">
              {busy && <CircleNotch className="size-4 animate-spin" />}
              {msg("settings.security.passkeys.create")}
            </Button>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
