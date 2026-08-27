"use client";

import * as React from "react";
import { toast } from "react-toastify";
import { CircleNotch, Plus, Trash, User } from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { Switch } from "@/shared/ui/primitives/switch";
import {
  createManagedAccount,
  deleteManagedAccount,
  getManagedAccounts,
  updateManagedAccountRole,
  type ManagedAccount,
} from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";

/** Render the administrator-owned local username and role directory. */
export function AdminAccountsSection() {
  const [accounts, setAccounts] = React.useState<ManagedAccount[]>([]);
  const [username, setUsername] = React.useState("");
  const [isAdmin, setIsAdmin] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [busyUsername, setBusyUsername] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    setLoading(true);
    try {
      setAccounts((await getManagedAccounts()).accounts);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  const createAccount = async () => {
    const normalized = username.trim().toLocaleLowerCase();
    if (!normalized) return;
    setBusyUsername(normalized);
    try {
      const created = await createManagedAccount({
        username: normalized,
        is_admin: isAdmin,
      });
      setAccounts((current) =>
        [...current.filter((account) => account.username !== created.username), created].sort(
          (left, right) => left.username.localeCompare(right.username),
        ),
      );
      setUsername("");
      setIsAdmin(false);
      toast.success(msg("settings.admin.accounts.created"));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyUsername(null);
    }
  };

  const setRole = async (account: ManagedAccount, next: boolean) => {
    setBusyUsername(account.username);
    try {
      const updated = await updateManagedAccountRole(account.username, next);
      setAccounts((current) =>
        current.map((entry) => (entry.username === updated.username ? updated : entry)),
      );
      toast.success(msg("settings.admin.accounts.role_saved"));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyUsername(null);
    }
  };

  const removeAccount = async (account: ManagedAccount) => {
    if (!window.confirm(msg("settings.admin.accounts.delete_confirm", { username: account.username }))) {
      return;
    }
    setBusyUsername(account.username);
    try {
      await deleteManagedAccount(account.username);
      setAccounts((current) => current.filter((entry) => entry.username !== account.username));
      toast.success(msg("settings.admin.accounts.deleted"));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyUsername(null);
    }
  };

  return (
    <section className="space-y-3 rounded-lg border border-border/50 p-3">
      <div className="flex items-center gap-2">
        <User className="size-4 text-muted-foreground" aria-hidden="true" />
        <div>
          <h3 className="text-sm font-semibold">{msg("settings.admin.accounts.title")}</h3>
          <p className="text-xs text-muted-foreground">
            {msg("settings.admin.accounts.description")}
          </p>
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto] sm:items-center">
        <Input
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          placeholder={msg("settings.admin.accounts.username_placeholder")}
          autoComplete="off"
          dir="ltr"
        />
        <label className="flex min-h-10 items-center justify-between gap-2 text-xs sm:justify-start">
          {msg("settings.admin.accounts.admin")}
          <Switch checked={isAdmin} onCheckedChange={setIsAdmin} />
        </label>
        <Button onClick={() => void createAccount()} disabled={!username.trim() || busyUsername !== null}>
          {busyUsername === username.trim().toLocaleLowerCase() ? (
            <CircleNotch className="size-4 animate-spin" />
          ) : (
            <Plus className="size-4" />
          )}
          {msg("settings.admin.accounts.create")}
        </Button>
      </div>

      {loading ? (
        <div className="flex justify-center py-6">
          <CircleNotch className="size-5 animate-spin text-muted-foreground" />
        </div>
      ) : accounts.length === 0 ? (
        <p className="py-4 text-center text-xs text-muted-foreground">
          {msg("settings.admin.accounts.empty")}
        </p>
      ) : (
        <div className="divide-y divide-border/40 overflow-hidden rounded-xl border border-border/50 bg-background/45">
          {accounts.map((account) => (
            <div
              key={account.username}
              className="flex flex-col gap-3 px-3.5 py-3.5 sm:flex-row sm:items-center sm:justify-between sm:gap-4"
            >
              <div className="flex min-w-0 items-center gap-3">
                <span className="flex size-9 shrink-0 items-center justify-center rounded-full border border-border/50 bg-muted/45 text-muted-foreground">
                  <User className="size-4" aria-hidden="true" />
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-foreground">
                    <bdi dir="ltr">{account.username}</bdi>
                  </p>
                  <p className="mt-0.5 truncate text-[0.6875rem] text-muted-foreground/80">
                    {account.adfs_seen
                      ? account.local_enabled
                        ? msg("settings.admin.accounts.source_adfs_local")
                        : msg("settings.admin.accounts.source_adfs")
                      : msg("settings.admin.accounts.source_local")}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-1.5 border-t border-border/35 pt-2.5 sm:border-t-0 sm:pt-0">
                <label className="flex min-h-[44px] flex-1 items-center justify-between gap-3 rounded-lg bg-muted/35 px-3 text-xs font-medium text-foreground transition-colors hover:bg-muted/55 sm:min-h-10 sm:flex-none">
                  {msg("settings.admin.accounts.admin_access")}
                  <Switch
                    checked={account.is_admin}
                    onCheckedChange={(next) => void setRole(account, next)}
                    disabled={busyUsername !== null}
                    aria-label={`${msg("settings.admin.accounts.admin")}: ${account.username}`}
                  />
                </label>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => void removeAccount(account)}
                  disabled={busyUsername !== null}
                  className="size-[44px] text-muted-foreground hover:bg-destructive/10 hover:text-destructive sm:size-9 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
                  aria-label={`${msg("settings.admin.accounts.delete")}: ${account.username}`}
                >
                  {busyUsername === account.username ? (
                    <CircleNotch className="size-3.5 animate-spin" />
                  ) : (
                    <Trash className="size-3.5" />
                  )}
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
