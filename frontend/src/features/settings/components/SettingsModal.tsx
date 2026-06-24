"use client";

import * as React from "react";
import { createPortal } from "react-dom";
import { signOut, useSession } from "next-auth/react";
import { toast } from "react-toastify";
import {
  BookOpen,
  Columns2,
  Check,
  Copy,
  KeyRound,
  ExternalLink,
  Feather,
  HardDrive,
  Keyboard,
  LogOut,
  Pencil,
  Plus,
  RotateCcw,
  Server,
  Shield,
  ShieldCheck,
  Sparkles,
  Table as TableIcon,
  Trash2,
  User,
  Info,
  X,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/primitives/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/ui/primitives/select";
import { Switch } from "@/shared/ui/primitives/switch";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { NumberInput } from "@/shared/ui/number-input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/ui/primitives/table";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/shared/ui/primitives/sheet";
import {
  ColumnHeader,
  ResetColumnsButton,
  type SortDir,
  useColumnFilters,
  useColumnResize,
} from "@/shared/ui/excel-filter";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { msg } from "@/shared/lib/messages";
import { formatStorageSize } from "@/shared/lib/formatters";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import {
  deleteStorageQuotaOverride,
  generateApiToken,
  getApiToken,
  getStorageQuotaOverrides,
  revokeApiToken,
  searchAdminUsers,
  setStorageQuotaOverride,
  type ApiTokenInfo,
  type DirectoryUserMatch,
  type StorageQuotaOverride,
} from "@/shared/lib/api";

import { useUserPrefs } from "../hooks/use-user-prefs";
import { useSettingsModal } from "../hooks/use-settings-modal";
import { ShortcutRecorder } from "./ShortcutRecorder";
import { SettingsRow } from "@/shared/ui/settings-row";

function WizardTab() {
  const { prefs, setPref } = useUserPrefs();

  return (
    <div className="space-y-1">
      <SettingsRow icon={Sparkles} label={msg("settings.wizard.code_assist.label")}>
        <Select
          value={prefs.wizardCodeAssist}
          onValueChange={(v) => setPref("wizardCodeAssist", v as typeof prefs.wizardCodeAssist)}
        >
          <SelectTrigger className="min-w-[160px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">{msg("settings.wizard.code_assist.auto")}</SelectItem>
            <SelectItem value="manual">{msg("settings.wizard.code_assist.manual")}</SelectItem>
          </SelectContent>
        </Select>
      </SettingsRow>

      <SettingsRow icon={Columns2} label={msg("settings.wizard.split_mode.label")}>
        <Select
          value={prefs.wizardSplitMode}
          onValueChange={(v) => setPref("wizardSplitMode", v as typeof prefs.wizardSplitMode)}
        >
          <SelectTrigger className="min-w-[160px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">{msg("settings.wizard.split_mode.auto")}</SelectItem>
            <SelectItem value="manual">{msg("settings.wizard.split_mode.manual")}</SelectItem>
          </SelectContent>
        </Select>
      </SettingsRow>
    </div>
  );
}

function AgentTab() {
  const { prefs, setPref } = useUserPrefs();

  return (
    <div className="space-y-1">
      <SettingsRow
        icon={Shield}
        label={msg("settings.agent.trust.label")}
        description={msg("settings.agent.trust.description")}
      >
        <Select
          value={prefs.agentTrustMode}
          onValueChange={(v) => setPref("agentTrustMode", v as typeof prefs.agentTrustMode)}
        >
          <SelectTrigger className="min-w-[160px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ask">{msg("settings.agent.trust.ask")}</SelectItem>
            <SelectItem value="auto_safe">{msg("settings.agent.trust.auto_safe")}</SelectItem>
            <SelectItem value="yolo">{msg("settings.agent.trust.yolo")}</SelectItem>
          </SelectContent>
        </Select>
      </SettingsRow>

      <SettingsRow
        icon={Keyboard}
        label={msg("settings.agent.shortcut.label")}
        description={msg("settings.agent.shortcut.description")}
      >
        <ShortcutRecorder />
      </SettingsRow>
    </div>
  );
}

function AccountTab() {
  const { data: session } = useSession();
  const { prefs, setPref } = useUserPrefs();
  const username = session?.user?.name ?? "";
  const role = (session?.user as Record<string, unknown> | undefined)?.role;
  const isAdmin = role === "admin";

  return (
    <div className="space-y-1">
      <SettingsRow icon={User} label={msg("settings.account.username.label")}>
        <span className="text-sm font-mono text-foreground" dir="ltr">
          {username || msg("settings.account.signed_out")}
        </span>
      </SettingsRow>

      <SettingsRow icon={ShieldCheck} label={msg("settings.account.role.label")}>
        <span className="text-xs uppercase tracking-wide font-semibold text-muted-foreground">
          {isAdmin ? msg("settings.account.role.admin") : msg("settings.account.role.user")}
        </span>
      </SettingsRow>

      <SettingsRow
        icon={Sparkles}
        label={msg("settings.account.advanced.label")}
        description={msg("settings.account.advanced.description")}
      >
        <Switch
          checked={prefs.advancedMode}
          onCheckedChange={(v) => setPref("advancedMode", v)}
        />
      </SettingsRow>

      <SettingsRow
        icon={Feather}
        label={msg("settings.account.lite.label")}
        description={msg("settings.account.lite.description")}
      >
        <Switch checked={prefs.liteMode} onCheckedChange={(v) => setPref("liteMode", v)} />
      </SettingsRow>

      <SettingsRow icon={LogOut} label={msg("settings.account.logout.label")}>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              size="icon-sm"
              onClick={() => signOut({ callbackUrl: "/login" })}
              disabled={!username}
              aria-label={msg("settings.account.logout.action")}
            >
              <LogOut className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{msg("settings.account.logout.action")}</TooltipContent>
        </Tooltip>
      </SettingsRow>
    </div>
  );
}

function UsernameCombobox({
  value,
  onChange,
  onSelect,
  disabled,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  onSelect: (entry: DirectoryUserMatch) => void;
  disabled?: boolean;
  placeholder?: string;
}) {
  const [results, setResults] = React.useState<DirectoryUserMatch[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [open, setOpen] = React.useState(false);
  const [pos, setPos] = React.useState<{ top: number; left: number; width: number; maxH: number } | null>(
    null,
  );
  const anchorRef = React.useRef<HTMLDivElement | null>(null);
  const debounceRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  React.useEffect(() => {
    const trimmed = value.trim();
    if (!trimmed) {
      setResults([]);
      setLoading(false);
      return;
    }
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setLoading(true);
    debounceRef.current = setTimeout(() => {
      searchAdminUsers(trimmed, 10)
        .then((data) => setResults(data.matches))
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    }, 250);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [value]);

  const dropdownOpen = open && value.trim().length > 0;

  // Render the suggestion list in a body portal with fixed positioning: the
  // cell sits inside the table's overflow-auto scroller, which clips an in-flow
  // absolute popup. Mirrors the column-filter dropdowns in this same table.
  React.useLayoutEffect(() => {
    if (!dropdownOpen) return;
    const updatePos = () => {
      const el = anchorRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const top = rect.bottom + 4;
      setPos({
        top,
        left: rect.left,
        width: rect.width,
        maxH: Math.max(120, window.innerHeight - top - 8),
      });
    };
    updatePos();
    window.addEventListener("scroll", updatePos, true);
    window.addEventListener("resize", updatePos);
    return () => {
      window.removeEventListener("scroll", updatePos, true);
      window.removeEventListener("resize", updatePos);
    };
  }, [dropdownOpen]);

  return (
    <div className="relative" ref={anchorRef}>
      <Input
        value={value}
        onChange={(event) => {
          onChange(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          // Defer so click on a suggestion can register before the popup unmounts.
          window.setTimeout(() => setOpen(false), 150);
        }}
        placeholder={placeholder}
        disabled={disabled}
        dir="ltr"
        // Suppress Chrome's native email-autofill popup: the field has its own
        // suggestion list, and the email-shaped placeholder otherwise triggers it.
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="none"
        spellCheck={false}
        data-1p-ignore
        data-lpignore="true"
        name="storage-quota-username-search"
        className="h-8 text-xs"
      />
      {dropdownOpen &&
        pos &&
        createPortal(
          <div
            // pointer-events-auto: the parent Sheet sets pointer-events:none on
            // <body>, which this body-portaled popup would otherwise inherit,
            // leaving the suggestions unclickable.
            className="pointer-events-auto fixed z-[9999] overflow-auto rounded-md border border-border/60 bg-background shadow-md"
            style={{
              top: pos.top,
              left: pos.left,
              width: pos.width,
              maxHeight: Math.min(192, pos.maxH),
            }}
          >
            {loading && results.length === 0 ? (
              <div className="px-3 py-2 text-xs text-muted-foreground">
                {msg("settings.admin.storage.searching")}
              </div>
            ) : results.length === 0 ? (
              <div className="px-3 py-2 text-xs text-muted-foreground">
                {msg("settings.admin.storage.no_suggestions")}
              </div>
            ) : (
              <ul role="listbox">
                {results.map((entry) => (
                  <li key={`${entry.source}:${entry.username}`}>
                    <button
                      type="button"
                      onMouseDown={(event) => {
                        event.preventDefault();
                        onSelect(entry);
                        setOpen(false);
                      }}
                      className="flex w-full items-center justify-between gap-2 px-3 py-1.5 text-start text-xs hover:bg-accent/50"
                      dir="ltr"
                    >
                      <span className="font-semibold text-foreground">{entry.username}</span>
                      {entry.source === "directory" && (
                        <span className="text-[0.6875rem] text-muted-foreground">
                          {msg("settings.admin.storage.source_directory")}
                        </span>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>,
          document.body,
        )}
    </div>
  );
}

const BYTES_PER_MB = 1024 * 1024;

function EditableBudgetCell({
  bytes,
  onSave,
  disabled,
}: {
  bytes: number;
  onSave: (nextBytes: number) => Promise<void>;
  disabled?: boolean;
}) {
  const toMb = React.useCallback((value: number) => String(Math.round(value / BYTES_PER_MB)), []);
  const [editing, setEditing] = React.useState(false);
  const [draft, setDraft] = React.useState<string>(toMb(bytes));
  const [saving, setSaving] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement | null>(null);

  React.useEffect(() => {
    if (!editing) setDraft(toMb(bytes));
  }, [bytes, editing, toMb]);

  React.useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const cancel = React.useCallback(() => {
    setDraft(toMb(bytes));
    setEditing(false);
  }, [bytes, toMb]);

  const commit = React.useCallback(async () => {
    const parsedMb = Number.parseInt(draft, 10);
    if (!Number.isFinite(parsedMb) || parsedMb < 1) {
      toast.error(msg("settings.admin.storage.budget_invalid"));
      return;
    }
    const nextBytes = parsedMb * BYTES_PER_MB;
    if (nextBytes === bytes) {
      setEditing(false);
      return;
    }
    setSaving(true);
    try {
      await onSave(nextBytes);
      setEditing(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("settings.admin.storage.save_failed"));
      cancel();
    } finally {
      setSaving(false);
    }
  }, [bytes, cancel, draft, onSave]);

  if (editing) {
    return (
      <span className="inline-flex items-center justify-center gap-1" dir="ltr">
        <input
          ref={inputRef}
          type="number"
          min={1}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void commit();
            } else if (event.key === "Escape") {
              event.preventDefault();
              cancel();
            }
          }}
          onBlur={cancel}
          disabled={saving}
          className="h-7 w-24 rounded-md border border-border/60 bg-background px-2 text-center text-xs tabular-nums outline-none focus:border-primary"
        />
        <span className="text-[0.6875rem] text-muted-foreground">MB</span>
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={() => !disabled && setEditing(true)}
      disabled={disabled}
      title={msg("settings.admin.storage.edit_hint")}
      className="group inline-flex items-center gap-1.5 rounded px-2 py-1 text-xs tabular-nums text-muted-foreground hover:bg-accent/40 hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
    >
      <span dir="ltr">{formatStorageSize(bytes)}</span>
      <Pencil className="size-3 opacity-0 transition group-hover:opacity-50" aria-hidden="true" />
    </button>
  );
}

function UsageMeter({ used, budget }: { used: number; budget: number }) {
  const pct = budget > 0 ? Math.min(100, (used / budget) * 100) : 0;
  const over = budget > 0 && used > budget;
  return (
    <div className="flex flex-col items-center gap-1" dir="ltr">
      <span
        className={`font-mono text-xs tabular-nums ${over ? "text-destructive" : "text-muted-foreground"}`}
      >
        {formatStorageSize(used)}
      </span>
      <div className="h-1 w-16 overflow-hidden rounded-full bg-[#E5DDD4]">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            over ? "bg-destructive" : "bg-[#3D2E22]/70"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function AdminTab() {
  const { data: session } = useSession();
  const isRtl = getActiveDir() === "rtl";
  const [overrides, setOverrides] = React.useState<StorageQuotaOverride[]>([]);
  const [defaultBytes, setDefaultBytes] = React.useState<number | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [tableOpen, setTableOpen] = React.useState(false);
  const [pendingUsername, setPendingUsername] = React.useState("");
  const [pendingBudgetMb, setPendingBudgetMb] = React.useState<number | "">("");
  const colFilters = useColumnFilters();
  const colResize = useColumnResize();
  const [sortKey, setSortKey] = React.useState<string>("username");
  const [sortDir, setSortDir] = React.useState<SortDir>("asc");

  const toggleSort = React.useCallback((key: string) => {
    setSortKey((prevKey) => {
      setSortDir((prevDir) => (prevKey === key ? (prevDir === "asc" ? "desc" : "asc") : "asc"));
      return key;
    });
  }, []);

  const filterOptions = React.useMemo(() => {
    const unique = (key: keyof StorageQuotaOverride) => {
      const vals = [
        ...new Set(overrides.map((o) => String(o[key] ?? "")).filter(Boolean)),
      ].sort();
      return vals.map((v) => ({ value: v, label: v }));
    };
    return {
      username: unique("username"),
      updated_by: unique("updated_by"),
    };
  }, [overrides]);

  const filteredOverrides = React.useMemo(() => {
    const items = overrides.filter((o) => {
      for (const [col, allowed] of Object.entries(colFilters.filters)) {
        const val = String((o as unknown as Record<string, unknown>)[col] ?? "");
        if (!allowed.has(val)) return false;
      }
      return true;
    });
    items.sort((a, b) => {
      const av = (a as unknown as Record<string, unknown>)[sortKey];
      const bv = (b as unknown as Record<string, unknown>)[sortKey];
      const aMissing = av == null || av === "";
      const bMissing = bv == null || bv === "";
      let cmp = 0;
      if (aMissing && bMissing) cmp = 0;
      else if (aMissing) cmp = -1;
      else if (bMissing) cmp = 1;
      else if (typeof av === "number" && typeof bv === "number") cmp = av - bv;
      else cmp = String(av).localeCompare(String(bv), "he", { numeric: true });
      return sortDir === "asc" ? cmp : -cmp;
    });
    return items;
  }, [overrides, colFilters.filters, sortKey, sortDir]);

  const loadOverrides = React.useCallback(async () => {
    if (!session?.backendAccessToken) return;
    setLoading(true);
    try {
      const data = await getStorageQuotaOverrides();
      setOverrides(data.overrides);
      setDefaultBytes(data.default_bytes);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [session?.backendAccessToken]);

  React.useEffect(() => {
    void loadOverrides();
  }, [loadOverrides]);

  React.useEffect(() => {
    if (!tableOpen) return;
    const id = setInterval(() => {
      void loadOverrides();
    }, 5000);
    return () => clearInterval(id);
  }, [tableOpen, loadOverrides]);

  const addPendingUser = React.useCallback(async () => {
    const normalizedUsername = pendingUsername.trim().toLowerCase();
    if (!normalizedUsername) {
      toast.error(msg("settings.admin.storage.username_required"));
      return;
    }
    if (pendingBudgetMb === "" || !Number.isFinite(pendingBudgetMb) || pendingBudgetMb < 1) {
      toast.error(msg("settings.admin.storage.budget_invalid"));
      return;
    }
    setBusy(true);
    try {
      const saved = await setStorageQuotaOverride(normalizedUsername, pendingBudgetMb * BYTES_PER_MB);
      setOverrides((prev) => {
        const without = prev.filter((row) => row.username !== saved.username);
        return [saved, ...without];
      });
      setPendingUsername("");
      setPendingBudgetMb("");
      toast.success(msg("settings.admin.storage.saved"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("settings.admin.storage.save_failed"));
    } finally {
      setBusy(false);
    }
  }, [pendingBudgetMb, pendingUsername]);

  const updateRowBudget = React.useCallback(
    async (targetUsername: string, nextBytes: number) => {
      const before = overrides;
      setOverrides((prev) =>
        prev.map((row) =>
          row.username === targetUsername
            ? { ...row, quota_bytes: nextBytes, effective_bytes: nextBytes }
            : row,
        ),
      );
      try {
        const saved = await setStorageQuotaOverride(targetUsername, nextBytes);
        setOverrides((prev) => prev.map((row) => (row.username === targetUsername ? saved : row)));
        toast.success(msg("settings.admin.storage.saved"));
      } catch (err) {
        setOverrides(before);
        throw err;
      }
    },
    [overrides],
  );

  const handleDelete = React.useCallback(
    async (targetUsername: string) => {
      const before = overrides;
      setOverrides((prev) => prev.filter((row) => row.username !== targetUsername));
      setBusy(true);
      try {
        await deleteStorageQuotaOverride(targetUsername);
        toast.success(msg("settings.admin.storage.deleted"));
      } catch (err) {
        setOverrides(before);
        toast.error(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [overrides],
  );

  const triggerLabel =
    overrides.length === 0
      ? msg("settings.admin.storage.view_list")
      : `${msg("settings.admin.storage.view_list")} (${overrides.length})`;

  return (
    <div className="space-y-4">
      {!session?.backendAccessToken && (
        <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          {msg("settings.admin.storage.auth_missing")}
        </div>
      )}

      <SettingsRow
        icon={HardDrive}
        label={msg("settings.admin.storage.title")}
        description={
          defaultBytes != null
            ? msg("settings.admin.storage.default_budget", { value: formatStorageSize(defaultBytes) })
            : undefined
        }
      >
        <span />
      </SettingsRow>

      <Sheet open={tableOpen} onOpenChange={setTableOpen}>
        <SheetTrigger asChild>
          <Button
            variant="outline"
            disabled={loading || !session?.backendAccessToken}
            className="w-full justify-center gap-2"
          >
            <TableIcon className="size-3.5" />
            <span>{triggerLabel}</span>
          </Button>
        </SheetTrigger>
        <SheetContent
          side={isRtl ? "left" : "right"}
          aria-describedby={undefined}
          className="w-full gap-0 p-0 sm:max-w-2xl"
        >
          <SheetHeader className="border-b border-border/40 px-6 py-4">
            <div className="flex items-center gap-2">
              <HardDrive className="size-4 text-muted-foreground" aria-hidden="true" />
              <SheetTitle>{msg("settings.admin.storage.title")}</SheetTitle>
            </div>
          </SheetHeader>

          <div className="flex items-center gap-3 border-b border-border/40 bg-muted/20 px-6 py-2">
            <span className="text-[0.6875rem] tabular-nums text-muted-foreground">
              {filteredOverrides.length === overrides.length
                ? overrides.length
                : `${filteredOverrides.length} / ${overrides.length}`}
            </span>
            {defaultBytes != null && (
              <span className="text-[0.6875rem] text-muted-foreground" dir="ltr">
                {msg("settings.admin.storage.default_budget", { value: formatStorageSize(defaultBytes) })}
              </span>
            )}
            <ResetColumnsButton resize={colResize} />
            {colFilters.activeCount > 0 && (
              <button
                type="button"
                onClick={colFilters.clearAll}
                className="text-[0.6875rem] text-muted-foreground hover:text-foreground cursor-pointer"
              >
                {msg("settings.admin.storage.clear_filters")}
              </button>
            )}
          </div>

          <div className="flex-1 overflow-auto">
            <div className="table-scroll">
              <Table style={{ minWidth: "560px" }}>
              <TableHeader className="sticky top-0 z-10 bg-muted/40 backdrop-blur-sm">
                <TableRow>
                  <ColumnHeader
                    label={msg("settings.admin.storage.username")}
                    sortKey="username"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    filterCol="username"
                    filterOptions={filterOptions.username}
                    filters={colFilters.filters}
                    onFilter={colFilters.setColumnFilter}
                    openFilter={colFilters.openFilter}
                    setOpenFilter={colFilters.setOpenFilter}
                    width={colResize.widths["username"]}
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("settings.admin.storage.budget")}
                    sortKey="effective_bytes"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    width={colResize.widths["effective_bytes"]}
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("settings.admin.storage.used")}
                    sortKey="used_bytes"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    width={colResize.widths["used_bytes"]}
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("settings.admin.storage.updated_by")}
                    sortKey="updated_by"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    filterCol="updated_by"
                    filterOptions={filterOptions.updated_by}
                    filters={colFilters.filters}
                    onFilter={colFilters.setColumnFilter}
                    openFilter={colFilters.openFilter}
                    setOpenFilter={colFilters.setOpenFilter}
                    width={colResize.widths["updated_by"]}
                    onResize={colResize.setColumnWidth}
                  />
                  <TableHead className="w-12" />
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow className="border-border/40 bg-muted/10">
                  <TableCell className="text-center" dir="ltr">
                    <UsernameCombobox
                      value={pendingUsername}
                      onChange={setPendingUsername}
                      onSelect={(entry) => setPendingUsername(entry.username)}
                      disabled={busy}
                      placeholder={msg("settings.admin.storage.username_placeholder")}
                    />
                  </TableCell>
                  <TableCell className="text-center">
                    <span className="inline-flex items-center justify-center gap-1" dir="ltr">
                      <NumberInput
                        value={pendingBudgetMb}
                        onChange={setPendingBudgetMb}
                        min={1}
                        disabled={busy}
                        className="mx-auto h-8 w-36"
                      />
                      <span className="text-[0.6875rem] text-muted-foreground">MB</span>
                    </span>
                  </TableCell>
                  <TableCell className="text-center text-xs text-muted-foreground/70">—</TableCell>
                  <TableCell className="text-center text-xs text-muted-foreground/70">—</TableCell>
                  <TableCell className="w-12 text-center">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          onClick={() => void addPendingUser()}
                          disabled={busy || !pendingUsername.trim() || pendingBudgetMb === ""}
                          aria-label={msg("settings.admin.storage.add_row")}
                        >
                          <Plus className="size-3.5 text-primary" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>{msg("settings.admin.storage.add_row")}</TooltipContent>
                    </Tooltip>
                  </TableCell>
                </TableRow>

                {filteredOverrides.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={5}
                      className="px-6 py-10 text-center text-sm text-muted-foreground"
                    >
                      {overrides.length === 0
                        ? msg("settings.admin.storage.empty")
                        : msg("settings.admin.storage.no_results")}
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredOverrides.map((item) => (
                    <TableRow
                      key={item.username}
                      className="border-border/40 hover:bg-accent/30"
                    >
                      <TableCell
                        className="max-w-[200px] truncate text-center font-semibold text-xs text-foreground"
                        dir="ltr"
                        title={item.username}
                      >
                        {item.username}
                      </TableCell>
                      <TableCell className="text-center">
                        <EditableBudgetCell
                          bytes={item.effective_bytes}
                          onSave={(nextBytes) => updateRowBudget(item.username, nextBytes)}
                          disabled={busy}
                        />
                      </TableCell>
                      <TableCell className="text-center">
                        <UsageMeter used={item.used_bytes} budget={item.effective_bytes} />
                      </TableCell>
                      <TableCell
                        className="max-w-[180px] truncate text-center text-xs text-muted-foreground"
                        dir="ltr"
                        title={item.updated_by || msg("settings.admin.storage.default")}
                      >
                        {item.updated_by || msg("settings.admin.storage.default")}
                      </TableCell>
                      <TableCell className="w-12 text-center">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              onClick={() => void handleDelete(item.username)}
                              disabled={busy}
                              className="close-button mx-auto disabled:opacity-50 disabled:cursor-not-allowed"
                              aria-label={msg("settings.admin.storage.delete")}
                            >
                              <X aria-hidden="true" />
                              <span className="sr-only">
                                {msg("settings.admin.storage.delete")}
                              </span>
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>
                            {msg("settings.admin.storage.delete")}
                          </TooltipContent>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
              </Table>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}

function AboutTab() {
  const { resetAll } = useUserPrefs();
  const { apiUrl, appVersion: version } = getRuntimeEnv();

  const handleResetAll = React.useCallback(() => {
    resetAll();
    toast.success(msg("settings.about.reset_all.success"));
  }, [resetAll]);

  return (
    <div className="space-y-1">
      <SettingsRow icon={Info} label={msg("settings.about.version.label")}>
        <span className="text-sm font-mono text-foreground" dir="ltr">
          {version}
        </span>
      </SettingsRow>

      <SettingsRow icon={Server} label={msg("settings.about.api_url.label")}>
        <span className="text-xs font-mono text-muted-foreground" dir="ltr">
          {apiUrl}
        </span>
      </SettingsRow>

      <SettingsRow
        icon={RotateCcw}
        label={msg("settings.about.reset_all.label")}
        description={msg("settings.about.reset_all.description")}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              size="icon-sm"
              onClick={handleResetAll}
              className="text-destructive hover:text-destructive"
              aria-label={msg("settings.about.reset_all.action")}
            >
              <RotateCcw className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{msg("settings.about.reset_all.action")}</TooltipContent>
        </Tooltip>
      </SettingsRow>
    </div>
  );
}

function ApiTab() {
  const { data: session } = useSession();
  const hasAuth = !!session?.backendAccessToken;
  const [info, setInfo] = React.useState<ApiTokenInfo | null>(null);
  const [loaded, setLoaded] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [revealed, setRevealed] = React.useState<string | null>(null);
  const [copied, setCopied] = React.useState(false);
  const [loadError, setLoadError] = React.useState<string | null>(null);

  // A failed passive load (most often the backend being unreachable) surfaces as
  // a calm inline banner rather than an error toast — the token panel is not
  // critical enough to interrupt, and a toast on tab-open reads as a bug. Toasts
  // stay reserved for the user-initiated generate/revoke actions below.
  const load = React.useCallback(async () => {
    if (!hasAuth) {
      setLoaded(true);
      return;
    }
    setLoadError(null);
    try {
      setInfo(await getApiToken());
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoaded(true);
    }
  }, [hasAuth]);

  React.useEffect(() => {
    void load();
  }, [load]);

  const handleGenerate = React.useCallback(async () => {
    setBusy(true);
    try {
      const created = await generateApiToken();
      setRevealed(created.token);
      setCopied(false);
      setInfo({ last4: created.last4, created_at: created.created_at, last_used_at: null });
      toast.success(msg("settings.api.generated_toast"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("settings.api.generate_failed"));
    } finally {
      setBusy(false);
    }
  }, []);

  const handleRevoke = React.useCallback(async () => {
    setBusy(true);
    try {
      await revokeApiToken();
      setInfo(null);
      setRevealed(null);
      toast.success(msg("settings.api.revoked_toast"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("settings.api.revoke_failed"));
    } finally {
      setBusy(false);
    }
  }, []);

  const handleCopy = React.useCallback(async () => {
    if (!revealed) return;
    try {
      await navigator.clipboard.writeText(revealed);
      setCopied(true);
    } catch {
      // Clipboard access can be blocked; the token stays visible to copy by hand.
    }
  }, [revealed]);

  const formatTimestamp = (iso: string) => new Date(iso).toLocaleString("he-IL");
  const docsUrl = `${getRuntimeEnv().apiUrl}/scalar`;

  if (!hasAuth) {
    return (
      <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
        {msg("settings.api.auth_missing")}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {loadError && (
        <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          {loadError}
        </div>
      )}

      <SettingsRow icon={KeyRound} label={msg("settings.api.title")}>
        {loaded &&
          !revealed &&
          (info ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon-sm"
                  variant="outline"
                  disabled={busy}
                  onClick={handleRevoke}
                  className="text-destructive hover:text-destructive"
                  aria-label={msg("settings.api.revoke")}
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{msg("settings.api.revoke")}</TooltipContent>
            </Tooltip>
          ) : (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon-sm"
                  disabled={busy}
                  onClick={handleGenerate}
                  aria-label={msg("settings.api.generate")}
                >
                  <KeyRound className="size-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{msg("settings.api.generate")}</TooltipContent>
            </Tooltip>
          ))}
      </SettingsRow>

      {revealed && (
        <div className="space-y-2 rounded-md border border-[#C8A882]/40 bg-[#FAF8F5] px-3 py-3">
          <div className="flex items-start gap-1.5 text-xs text-[#7A1E13]">
            <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
            <span>{msg("settings.api.reveal_warning")}</span>
          </div>
          <div
            dir="ltr"
            className="flex items-center justify-between gap-2 rounded bg-[#3D2E22]/5 px-2 py-1.5"
          >
            <code className="min-w-0 flex-1 break-all font-mono text-xs text-[#3D2E22]">
              {revealed}
            </code>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon-sm"
                  variant="outline"
                  className="shrink-0"
                  onClick={handleCopy}
                  aria-label={copied ? msg("settings.api.copied") : msg("settings.api.copy")}
                >
                  {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {copied ? msg("settings.api.copied") : msg("settings.api.copy")}
              </TooltipContent>
            </Tooltip>
          </div>
          <button
            type="button"
            onClick={() => setRevealed(null)}
            className="group relative inline-flex w-full cursor-pointer rounded-lg bg-muted p-1 transform-gpu transition-transform duration-75 ease-out active:scale-[0.97] focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
          >
            <span className="flex-1 rounded-md bg-background px-4 py-2.5 text-center text-sm font-medium text-foreground shadow-sm transition-[box-shadow,transform] duration-150 ease-out group-hover:-translate-y-px group-hover:shadow-md">
              {msg("settings.api.done")}
            </span>
          </button>
        </div>
      )}

      {loaded && !revealed && info && (
        <div className="space-y-2 rounded-md border border-border/50 px-3 py-3 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="text-muted-foreground">{msg("settings.api.active_label")}</span>
            <code dir="ltr" className="font-mono text-foreground">
              {msg("settings.api.token_masked", { last4: info.last4 })}
            </code>
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="text-muted-foreground">{msg("settings.api.created")}</span>
            <span dir="ltr">{formatTimestamp(info.created_at)}</span>
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="text-muted-foreground">{msg("settings.api.last_used")}</span>
            <span dir="ltr">
              {info.last_used_at
                ? formatTimestamp(info.last_used_at)
                : msg("settings.api.never_used")}
            </span>
          </div>
        </div>
      )}

      {loaded && !revealed && !info && !loadError && (
        <p className="text-xs text-muted-foreground">{msg("settings.api.none")}</p>
      )}

      <SettingsRow
        icon={BookOpen}
        label={msg("settings.api.docs_label")}
        description={msg("settings.api.docs_description")}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              size="icon-sm"
              asChild
              aria-label={msg("settings.api.docs_action")}
            >
              <a href={docsUrl} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="size-3.5" />
              </a>
            </Button>
          </TooltipTrigger>
          <TooltipContent>{msg("settings.api.docs_action")}</TooltipContent>
        </Tooltip>
      </SettingsRow>
    </div>
  );
}


const SETTINGS_TAB_ORDER = [
  "wizard",
  "agent",
  "admin",
  "account",
  "api",
  "about",
] as const;
type SettingsTab = (typeof SETTINGS_TAB_ORDER)[number];

const SETTINGS_TAB_TRIGGER_CLASS =
  "relative z-10 w-full shrink-0 whitespace-nowrap text-center text-[clamp(0.75rem,2.2vw,0.875rem)] rounded-md px-1.5 py-2 font-medium cursor-pointer border-none shadow-none bg-transparent data-[state=active]:bg-transparent data-[state=active]:text-white data-[state=active]:shadow-none data-[state=active]:border-none gap-1.5 leading-tight";

export function SettingsModal() {
  const { open, setOpen } = useSettingsModal();
  const { data: session } = useSession();
  const isAdmin = session?.user?.role === "admin";
  const [activeTab, setActiveTab] = React.useState<SettingsTab>("wizard");
  const tabs = React.useMemo(
    () => SETTINGS_TAB_ORDER.filter((tab) => isAdmin || tab !== "admin"),
    [isAdmin],
  );
  React.useEffect(() => {
    if (!tabs.includes(activeTab)) setActiveTab("wizard");
  }, [activeTab, tabs]);
  const tabCount = tabs.length;
  const listElRef = React.useRef<HTMLElement | null>(null);
  const observerRef = React.useRef<ResizeObserver | null>(null);
  const [indicator, setIndicator] = React.useState<{ left: number; width: number } | null>(null);

  // Measure the active trigger from the DOM rather than computing from its
  // index: the columns are `minmax(max-content, 1fr)`, so they are equal-width
  // only when every label fits the equal share. On narrow widths — and when the
  // admin tab drops the count to five — the columns diverge, and an index-based
  // offset slides the pill off the active tab. offsetLeft/Width are physical, so
  // this stays correct in RTL too. The >0 guard avoids painting a collapsed pill
  // before the dialog grid has laid out.
  const measure = React.useCallback(() => {
    const active = listElRef.current?.querySelector<HTMLElement>('[data-state="active"]');
    if (active && active.offsetWidth > 0) {
      setIndicator({ left: active.offsetLeft, width: active.offsetWidth });
    }
  }, []);

  // Callback ref on the pill: fires when the dialog content is actually
  // attached. Radix portals the content a beat after `open` flips, so a parent
  // open-keyed effect races the mount and can bind to a null ref; a callback ref
  // can't. The ResizeObserver delivers the first valid measure (0→full as the
  // dialog lays out) and keeps the pill aligned on responsive resize.
  const bindIndicator = React.useCallback(
    (node: HTMLDivElement | null) => {
      observerRef.current?.disconnect();
      observerRef.current = null;
      const list = node?.parentElement ?? null;
      listElRef.current = list;
      if (!list) return;
      measure();
      const observer = new ResizeObserver(measure);
      observer.observe(list);
      observerRef.current = observer;
    },
    [measure],
  );

  // Switching tab resizes nothing, so the observer won't fire — re-measure here
  // so the pill slides to the newly active tab.
  React.useLayoutEffect(() => {
    measure();
  }, [activeTab, measure]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-w-2xl sm:max-w-2xl p-0 overflow-hidden">
        <DialogHeader className="px-6 pt-6 pb-4 border-b border-border/40">
          <DialogTitle>{msg("settings.title")}</DialogTitle>
          <DialogDescription>{msg("settings.subtitle")}</DialogDescription>
        </DialogHeader>

        <Tabs
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as SettingsTab)}
          className="px-6 pb-6 pt-2"
        >
          <TabsList
            className="relative grid w-full rounded-lg bg-muted p-1 gap-1 border-none shadow-none h-auto items-stretch overflow-x-auto no-scrollbar"
            style={{ gridTemplateColumns: `repeat(${tabCount}, minmax(max-content, 1fr))` }}
          >
            <div
              ref={bindIndicator}
              aria-hidden="true"
              className={`pointer-events-none absolute top-1 bottom-1 rounded-md bg-[#3D2E22] shadow-sm transition-[left,width] duration-200 ease-[cubic-bezier(0.2,0.8,0.2,1)] motion-reduce:transition-none ${
                indicator ? "opacity-100" : "opacity-0"
              }`}
              style={indicator ? { left: indicator.left, width: indicator.width } : undefined}
            />
            <TabsTrigger value="wizard" className={SETTINGS_TAB_TRIGGER_CLASS}>
              {msg("settings.tab.wizard")}
            </TabsTrigger>
            <TabsTrigger value="agent" className={SETTINGS_TAB_TRIGGER_CLASS}>
              {msg("settings.tab.agent")}
            </TabsTrigger>
            {isAdmin && (
              <TabsTrigger value="admin" className={SETTINGS_TAB_TRIGGER_CLASS}>
                {msg("settings.tab.admin")}
              </TabsTrigger>
            )}
            <TabsTrigger value="account" className={SETTINGS_TAB_TRIGGER_CLASS}>
              {msg("settings.tab.account")}
            </TabsTrigger>
            <TabsTrigger value="api" className={SETTINGS_TAB_TRIGGER_CLASS}>
              {msg("settings.tab.api")}
            </TabsTrigger>
            <TabsTrigger value="about" className={SETTINGS_TAB_TRIGGER_CLASS}>
              {msg("settings.tab.about")}
            </TabsTrigger>
          </TabsList>

          <div className="mt-4 max-h-[60vh] overflow-y-auto pr-1">
            <TabsContent value="wizard">
              <WizardTab />
            </TabsContent>
            <TabsContent value="agent">
              <AgentTab />
            </TabsContent>
            {isAdmin && (
              <TabsContent value="admin">
                <AdminTab />
              </TabsContent>
            )}
            <TabsContent value="account">
              <AccountTab />
            </TabsContent>
            <TabsContent value="api">
              <ApiTab />
            </TabsContent>
            <TabsContent value="about">
              <AboutTab />
            </TabsContent>
          </div>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
