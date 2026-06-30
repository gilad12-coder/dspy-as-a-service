"use client";

import * as React from "react";
import { Check, KeyRound, Loader2, Pencil, Trash2, X } from "lucide-react";
import { toast } from "react-toastify";
import { msg, formatMsg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { useByokKeys } from "../providers/byok-provider";
import { formatResetDate } from "../lib/credit";
import { BYOK_PROVIDERS, type ByokProviderInfo, type KeyStatus } from "../lib/byok";
import { ProviderLogo } from "./ProviderLogo";
import { ByokJsonImport } from "./ByokJsonImport";

/** The status pill next to a saved key. Gold for verified, calm muted/destructive otherwise. */
function StatusPill({ status }: { status: KeyStatus }) {
  const map: Record<KeyStatus, { label: string; className: string }> = {
    verified: {
      label: msg("settings.keys.verified"),
      className: "bg-[#C8A882]/15 text-[#8a6d44]",
    },
    unverified: {
      label: msg("settings.keys.unverified"),
      className: "bg-muted text-muted-foreground",
    },
    invalid: {
      label: msg("settings.keys.invalid"),
      className: "bg-destructive/10 text-destructive",
    },
  };
  const { label, className } = map[status];
  return (
    <span className={cn("rounded-full px-2 py-0.5 text-[0.6875rem] font-medium", className)}>
      {label}
    </span>
  );
}

function ProviderKeyRow({ provider }: { provider: ByokProviderInfo }) {
  const { keyFor, saveKey, verifyKey, removeKey } = useByokKeys();
  const { locale } = useLocale();
  const saved = keyFor(provider.slug);

  const [editing, setEditing] = React.useState(false);
  const [secret, setSecret] = React.useState("");
  const [baseUrl, setBaseUrl] = React.useState("");
  const [saving, setSaving] = React.useState(false);
  const [verifying, setVerifying] = React.useState(false);

  const startEditing = () => {
    setSecret("");
    setBaseUrl(saved?.apiBase ?? "");
    setEditing(true);
  };

  const handleSave = async () => {
    const trimmed = secret.trim();
    if (!trimmed) return;
    setSaving(true);
    try {
      const status = await saveKey(provider.slug, trimmed, { apiBase: baseUrl.trim() || null });
      setSecret("");
      setBaseUrl("");
      setEditing(false);
      // The vault verifies on entry, so a saved key can already come back
      // rejected; surface that honestly rather than a blanket "saved".
      if (status === "invalid") {
        toast.error(msg("settings.keys.invalid_toast"));
      } else {
        toast.success(msg("settings.keys.saved_toast"));
      }
    } catch {
      toast.error(msg("settings.keys.save_failed_toast"));
    } finally {
      setSaving(false);
    }
  };

  const handleVerify = async () => {
    setVerifying(true);
    try {
      const status = await verifyKey(provider.slug);
      if (status === "verified") {
        toast.success(msg("settings.keys.verified_toast"));
      } else if (status === "invalid") {
        toast.error(msg("settings.keys.invalid_toast"));
      } else {
        toast.info(msg("settings.keys.unverified_toast"));
      }
    } catch {
      toast.error(msg("settings.keys.verify_failed_toast"));
    } finally {
      setVerifying(false);
    }
  };

  const handleRemove = async () => {
    try {
      await removeKey(provider.slug);
      toast.success(msg("settings.keys.removed_toast"));
    } catch {
      toast.error(msg("settings.keys.remove_failed_toast"));
    }
  };

  return (
    <div className="rounded-lg border border-border/50 px-3 py-2.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <ProviderLogo slug={provider.slug} size={28} />
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="text-sm font-medium text-foreground">{provider.label}</span>
            {saved && (
              <span className="flex items-center gap-2">
                <code dir="ltr" className="font-mono text-xs text-muted-foreground">
                  ••••&nbsp;{saved.last4}
                </code>
                <StatusPill status={saved.status} />
              </span>
            )}
            {saved?.apiBase && (
              <code dir="ltr" className="truncate font-mono text-[0.6875rem] text-muted-foreground/70">
                {saved.apiBase}
              </code>
            )}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {!saved && !editing && (
            <Button variant="outline" size="sm" onClick={startEditing}>
              <KeyRound className="size-3.5" />
              {msg("settings.keys.add")}
            </Button>
          )}
          {saved && saved.status !== "verified" && !editing && (
            <Button variant="outline" size="sm" disabled={verifying} onClick={handleVerify}>
              {verifying ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Check className="size-3.5" />
              )}
              {verifying ? msg("settings.keys.verifying") : msg("settings.keys.verify")}
            </Button>
          )}
          {saved && !editing && (
            <>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon-sm"
                    onClick={startEditing}
                    aria-label={msg("settings.keys.replace")}
                  >
                    <Pencil className="size-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{msg("settings.keys.replace")}</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon-sm"
                    onClick={handleRemove}
                    className="text-destructive hover:text-destructive"
                    aria-label={msg("settings.keys.remove")}
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{msg("settings.keys.remove")}</TooltipContent>
              </Tooltip>
            </>
          )}
        </div>
      </div>

      {saved && !editing && (
        <p className="mt-1.5 text-[0.6875rem] text-muted-foreground/70">
          {formatMsg("settings.keys.added", { date: formatResetDate(saved.addedAt, locale) })}
        </p>
      )}

      {editing && (
        <div className="mt-2.5 flex flex-col gap-2 animate-in fade-in-0 slide-in-from-top-1">
          <div className="flex items-center gap-2">
            <Input
              dir="ltr"
              type="password"
              autoFocus
              autoComplete="new-password"
              placeholder={provider.placeholder}
              value={secret}
              onChange={(e) => setSecret(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleSave();
                if (e.key === "Escape") setEditing(false);
              }}
              className="h-8 flex-1"
            />
            <Button size="sm" onClick={handleSave} disabled={!secret.trim() || saving}>
              {saving ? <Loader2 className="size-3.5 animate-spin" /> : msg("settings.keys.save")}
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setEditing(false)}
              aria-label={msg("settings.keys.cancel")}
            >
              <X className="size-3.5" />
            </Button>
          </div>
          <Input
            dir="ltr"
            type="url"
            autoComplete="off"
            placeholder={msg("settings.keys.base_url_placeholder")}
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleSave();
              if (e.key === "Escape") setEditing(false);
            }}
            className="h-7 text-xs"
          />
          <p className="text-[0.6875rem] text-muted-foreground/70">
            {msg("settings.keys.base_url_hint")}
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * BYOK provider-key manager — lives in the Settings "Providers" tab.
 *
 * Each major provider gets a row: add a key, verify it, replace it, or remove
 * it. The secret is entered once and never shown again (only its masked tail),
 * and the privacy line states the encrypt-at-rest guarantee up front. This is
 * the in-app home referenced by the model picker's BYOK mode.
 */
export function ByokKeysSection() {
  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <KeyRound className="size-4 text-muted-foreground" aria-hidden="true" />
          <span className="text-sm font-semibold text-foreground">
            {msg("settings.keys.title")}
          </span>
        </div>
        <p className="text-xs text-muted-foreground">{msg("settings.keys.description")}</p>
      </div>

      <div className="flex flex-col gap-2">
        {BYOK_PROVIDERS.map((p) => (
          <ProviderKeyRow key={p.slug} provider={p} />
        ))}
      </div>

      <ByokJsonImport />
    </div>
  );
}
