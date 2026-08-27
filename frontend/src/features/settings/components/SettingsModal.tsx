"use client";

import * as React from "react";
import { useSession } from "next-auth/react";
import { toast } from "react-toastify";
import { motion, useReducedMotion } from "framer-motion";
import {
  ChatText,
  BookOpen,
  CircleNotch,
  Robot,
  Brain,
  Columns,
  Cpu,
  Key,
  ArrowSquareOut,
  Feather,
  HardDrive,
  Keyboard,
  type Icon,
  Microphone,
  PencilSimple,
  PencilSimpleLine,
  Plug,
  Plus,
  ArrowCounterClockwise,
  HardDrives,
  Shield,
  FadersHorizontal,
  ShieldCheck,
  Sparkle,
  Table as TableIcon,
  Tag,
  MagnifyingGlass,
  Trash,
  User,
  Info,
  X,
} from "@/shared/ui/icons";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/primitives/tabs";
import { track, TelemetryEvent } from "@/shared/lib/telemetry";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/ui/primitives/select";
import { Switch } from "@/shared/ui/primitives/switch";
import { Button } from "@/shared/ui/primitives/button";
import { CopyButton } from "@/shared/ui/copy-button";
import { ByokKeysSection } from "@/features/byok";
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
  ResetFiltersButton,
  type SortDir,
  useColumnFilters,
  useColumnResize,
} from "@/shared/ui/excel-filter";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { ExportTableMenu } from "@/shared/ui/export-table-menu";
import { msg } from "@/shared/lib/messages";
import { formatStorageSize } from "@/shared/lib/formatters";
import { cachedCatalog, getModelCatalog } from "@/shared/lib/model-catalog";
import type { CatalogModel } from "@/shared/types/api";
import { ModelChip } from "@/shared/ui/model-chip";
import { ModelConfigModal, useRecentModelConfigs } from "@/features/submit";
import { getActiveDir, getActiveIntlLocale } from "@/shared/lib/runtime-locale";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { useTutorialContext } from "@/features/tutorial";
import {
  createManagedAccount,
  deleteManagedAccount,
  deleteStorageQuotaOverride,
  generateApiToken,
  getApiToken,
  getManagedAccounts,
  getMemorySettings,
  revokeApiToken,
  setStorageQuotaOverride,
  updateManagedAccountRole,
  updateMemorySettings,
  type ApiTokenInfo,
  type ManagedAccount,
  type MemoryKnob,
  type MemoryKnobName,
  type MemorySettings,
} from "@/shared/lib/api";

import { useUserPrefs } from "../hooks/use-user-prefs";
import { useSettingsModal } from "../hooks/use-settings-modal";
import { useIsPhone } from "@/shared/hooks/use-device-class";
import { isPhoneSettingsTab } from "@/shared/lib/device-class";
import { ShortcutRecorder } from "./ShortcutRecorder";
import { SettingsRow } from "@/shared/ui/settings-row";

function WizardTab() {
  const { prefs, setPref } = useUserPrefs();

  return (
    <div className="space-y-1">
      <SettingsRow icon={Sparkle} label={msg("settings.wizard.code_assist.label")}>
        <Select
          value={prefs.wizardCodeAssist}
          onValueChange={(v) => setPref("wizardCodeAssist", v as typeof prefs.wizardCodeAssist)}
        >
          <SelectTrigger className="h-[44px] w-full min-w-0 sm:h-8 sm:w-auto sm:min-w-[160px] [@media(hover:none)_and_(pointer:coarse)]:h-[44px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">{msg("settings.wizard.code_assist.auto")}</SelectItem>
            <SelectItem value="manual">{msg("settings.wizard.code_assist.manual")}</SelectItem>
          </SelectContent>
        </Select>
      </SettingsRow>

      <SettingsRow icon={Columns} label={msg("settings.wizard.split_mode.label")}>
        <Select
          value={prefs.wizardSplitMode}
          onValueChange={(v) => setPref("wizardSplitMode", v as typeof prefs.wizardSplitMode)}
        >
          <SelectTrigger className="h-[44px] w-full min-w-0 sm:h-8 sm:w-auto sm:min-w-[160px] [@media(hover:none)_and_(pointer:coarse)]:h-[44px]">
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

function TaggingTab() {
  const { prefs, setPref } = useUserPrefs();
  const [modelDialogOpen, setModelDialogOpen] = React.useState(false);
  // The same recents the submit wizard's model dialog keeps — one shared
  // localStorage list across every model-config surface.
  const { recentConfigs, saveToRecent, removeRecentConfig } = useRecentModelConfigs();
  // Same managed-catalog source the tagger setup feeds the dialog: thinking
  // detection and the chip's vision badge need the model metadata.
  const [catalogModels, setCatalogModels] = React.useState<CatalogModel[] | null>(
    cachedCatalog()?.models ?? null,
  );
  React.useEffect(() => {
    if (catalogModels) return;
    let cancelled = false;
    getModelCatalog()
      .then((c) => {
        if (!cancelled) setCatalogModels(c.models);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [catalogModels]);

  return (
    <div className="space-y-1">
      <SettingsRow
        icon={Tag}
        label={msg("settings.tagger.assist.label")}
        description={msg("settings.tagger.assist.description")}
      >
        <Switch checked={prefs.taggerAssist} onCheckedChange={(v) => setPref("taggerAssist", v)} />
      </SettingsRow>

      <SettingsRow
        icon={Cpu}
        label={msg("settings.tagger.default_model.label")}
        description={msg("settings.tagger.default_model.description")}
      >
        <ModelChip
          config={prefs.taggerAssistModel}
          emptyLabel={msg("tagger.assist.model.placeholder")}
          catalogModels={catalogModels ?? undefined}
          onClick={() => setModelDialogOpen(true)}
          onRemove={
            prefs.taggerAssistModel.name
              ? () => setPref("taggerAssistModel", { name: "" })
              : undefined
          }
        />
      </SettingsRow>
      <ModelConfigModal
        open={modelDialogOpen}
        onOpenChange={setModelDialogOpen}
        config={prefs.taggerAssistModel}
        onSave={(cfg) => {
          saveToRecent(cfg);
          setPref("taggerAssistModel", cfg);
        }}
        roleLabel={msg("settings.tagger.default_model.label")}
        recentConfigs={recentConfigs}
        onRemoveRecent={removeRecentConfig}
      />
    </div>
  );
}

// One agent-memory knob: the stepper plus a reset affordance that appears only
// while the value overrides the tool default (OptMem's "commented line means:
// follow the tool" semantics, inverted into UI).
function MemoryKnobControl({
  knob,
  step,
  onCommit,
}: {
  knob: MemoryKnob;
  step: number;
  onCommit: (value: number | null) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      {knob.override != null && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => onCommit(null)}
              className="size-[44px] sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
              aria-label={msg("settings.agent.memory.reset")}
            >
              <ArrowCounterClockwise className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{msg("settings.agent.memory.reset")}</TooltipContent>
        </Tooltip>
      )}
      <NumberInput
        value={knob.value}
        onChange={onCommit}
        min={knob.min}
        max={knob.max}
        step={step}
        className="w-[132px]"
      />
    </div>
  );
}

function AgentTab() {
  const { prefs, setPref } = useUserPrefs();
  const [memory, setMemory] = React.useState<MemorySettings | null>(null);
  const transcriptionEnabled = getRuntimeEnv().transcriptionEnabled;
  const saveTimers = React.useRef<Partial<Record<MemoryKnobName, ReturnType<typeof setTimeout>>>>(
    {},
  );

  React.useEffect(() => {
    let cancelled = false;
    getMemorySettings()
      .then((s) => {
        if (!cancelled) setMemory(s);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // Optimistic local update, then a debounced PUT: NumberInput commits every
  // keystroke and stepper tap, and each would otherwise be a round-trip.
  const commitKnob = React.useCallback((name: MemoryKnobName, value: number | null) => {
    setMemory((prev) =>
      prev
        ? {
            ...prev,
            [name]: { ...prev[name], value: value ?? prev[name].default, override: value },
          }
        : prev,
    );
    const timers = saveTimers.current;
    const pending = timers[name];
    if (pending) clearTimeout(pending);
    timers[name] = setTimeout(() => {
      updateMemorySettings({ [name]: value })
        .then((s) => {
          setMemory(s);
          toast.success(msg("settings.saved"), { autoClose: 1500, toastId: "settings-saved" });
        })
        .catch(() => {
          toast.error(msg("settings.agent.memory.save_failed"), {
            toastId: "memory-save-failed",
          });
          getMemorySettings()
            .then(setMemory)
            .catch(() => {});
        });
    }, 600);
  }, []);

  return (
    <div className="space-y-1">
      {transcriptionEnabled && (
        <SettingsRow
          icon={Microphone}
          label={msg("settings.agent.dictation.label")}
          description={msg("settings.agent.dictation.description")}
        >
          <Switch
            checked={prefs.dictationEnabled}
            onCheckedChange={(v) => setPref("dictationEnabled", v)}
          />
        </SettingsRow>
      )}

      <SettingsRow
        icon={Shield}
        label={msg("settings.agent.trust.label")}
        description={msg("settings.agent.trust.description")}
      >
        <Select
          value={prefs.agentTrustMode}
          onValueChange={(v) => setPref("agentTrustMode", v as typeof prefs.agentTrustMode)}
        >
          <SelectTrigger className="h-[44px] w-full min-w-0 sm:h-8 sm:w-auto sm:min-w-[160px] [@media(hover:none)_and_(pointer:coarse)]:h-[44px]">
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

      {memory && (
        <>
          <SettingsRow
            icon={Brain}
            label={msg("settings.agent.memory.wake.label")}
            description={msg("settings.agent.memory.wake.description")}
          >
            <MemoryKnobControl
              knob={memory.wake_lines}
              step={8}
              onCommit={(v) => commitKnob("wake_lines", v)}
            />
          </SettingsRow>

          <SettingsRow
            icon={PencilSimpleLine}
            label={msg("settings.agent.memory.entry.label")}
            description={msg("settings.agent.memory.entry.description")}
          >
            <MemoryKnobControl
              knob={memory.entry_chars}
              step={20}
              onCommit={(v) => commitKnob("entry_chars", v)}
            />
          </SettingsRow>

          <SettingsRow
            icon={MagnifyingGlass}
            label={msg("settings.agent.memory.recall.label")}
            description={msg("settings.agent.memory.recall.description")}
          >
            <MemoryKnobControl
              knob={memory.recall_chars}
              step={500}
              onCommit={(v) => commitKnob("recall_chars", v)}
            />
          </SettingsRow>
        </>
      )}
    </div>
  );
}

function AccountTab() {
  const { data: session } = useSession();
  const { prefs, setPref } = useUserPrefs();
  // Advanced/expand/lite toggles shape the wizard and the desktop sidebar,
  // neither of which exists in the phone shell.
  const isPhone = useIsPhone();
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

      {!isPhone && (
        <>
          <SettingsRow
            icon={FadersHorizontal}
            label={msg("settings.account.advanced_mode.label")}
            description={msg("settings.account.advanced_mode.description")}
          >
            <Switch
              checked={prefs.advancedMode}
              onCheckedChange={(v) => setPref("advancedMode", v)}
            />
          </SettingsRow>

          <SettingsRow
            icon={Sparkle}
            label={msg("settings.account.expand_advanced.label")}
            description={msg("settings.account.expand_advanced.description")}
          >
            <Switch
              checked={prefs.expandAdvanced}
              onCheckedChange={(v) => setPref("expandAdvanced", v)}
            />
          </SettingsRow>

          <SettingsRow
            icon={Feather}
            label={msg("settings.account.lite.label")}
            description={msg("settings.account.lite.description")}
          >
            <Switch checked={prefs.liteMode} onCheckedChange={(v) => setPref("liteMode", v)} />
          </SettingsRow>
        </>
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
  const toMb = React.useCallback((value: number) => Math.round(value / BYTES_PER_MB), []);
  const [editing, setEditing] = React.useState(false);
  const [draft, setDraft] = React.useState<number>(toMb(bytes));
  const [saving, setSaving] = React.useState(false);
  const wrapperRef = React.useRef<HTMLSpanElement | null>(null);

  React.useEffect(() => {
    if (!editing) setDraft(toMb(bytes));
  }, [bytes, editing, toMb]);

  React.useEffect(() => {
    if (editing) wrapperRef.current?.querySelector("input")?.select();
  }, [editing]);

  const cancel = React.useCallback(() => {
    setDraft(toMb(bytes));
    setEditing(false);
  }, [bytes, toMb]);

  const commit = React.useCallback(async () => {
    if (!Number.isFinite(draft) || draft < 1) {
      toast.error(msg("settings.admin.storage.budget_invalid"));
      return;
    }
    const nextBytes = draft * BYTES_PER_MB;
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
      <span
        ref={wrapperRef}
        className="inline-flex items-center justify-center gap-1"
        dir="ltr"
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            void commit();
          } else if (event.key === "Escape") {
            // stopPropagation keeps Radix from closing the whole sheet.
            event.preventDefault();
            event.stopPropagation();
            cancel();
          }
        }}
        onMouseDown={(event) => {
          // Keep focus in the field while the +/- steppers are clicked, so the
          // blur-cancel below never fires mid-adjustment (Safari leaves
          // relatedTarget null for button clicks).
          if ((event.target as HTMLElement).closest("button")) event.preventDefault();
        }}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) cancel();
        }}
      >
        <NumberInput value={draft} onChange={setDraft} min={1} disabled={saving} className="h-8 w-36" />
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
      <PencilSimple
        className="size-3 opacity-0 transition group-hover:opacity-50"
        aria-hidden="true"
      />
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
  const [accounts, setAccounts] = React.useState<ManagedAccount[]>([]);
  const [defaultBytes, setDefaultBytes] = React.useState<number | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [busyUsername, setBusyUsername] = React.useState<string | null>(null);
  const [tableOpen, setTableOpen] = React.useState(false);
  const [newUsername, setNewUsername] = React.useState("");
  const [newIsAdmin, setNewIsAdmin] = React.useState(false);
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
    const unique = (key: keyof ManagedAccount) => {
      const vals = [...new Set(accounts.map((a) => String(a[key] ?? "")).filter(Boolean))].sort();
      return vals.map((v) => ({ value: v, label: v }));
    };
    return {
      username: unique("username"),
      quota_updated_by: unique("quota_updated_by"),
    };
  }, [accounts]);

  const filteredAccounts = React.useMemo(() => {
    const items = accounts.filter((a) => {
      for (const [col, allowed] of Object.entries(colFilters.filters)) {
        const val = String((a as unknown as Record<string, unknown>)[col] ?? "");
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
  }, [accounts, colFilters.filters, sortKey, sortDir]);

  const loadAccounts = React.useCallback(async () => {
    if (!session?.backendAccessToken) return;
    setLoading(true);
    try {
      const data = await getManagedAccounts();
      setAccounts(data.accounts);
      setDefaultBytes(data.default_bytes);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [session?.backendAccessToken]);

  React.useEffect(() => {
    void loadAccounts();
  }, [loadAccounts]);

  React.useEffect(() => {
    if (!tableOpen) return;
    const id = setInterval(() => {
      void loadAccounts();
    }, 5000);
    return () => clearInterval(id);
  }, [tableOpen, loadAccounts]);

  const createAccount = React.useCallback(async () => {
    const normalized = newUsername.trim().toLocaleLowerCase();
    if (!normalized) return;
    setBusyUsername(normalized);
    try {
      const created = await createManagedAccount({ username: normalized, is_admin: newIsAdmin });
      setAccounts((current) =>
        [...current.filter((account) => account.username !== created.username), created].sort(
          (left, right) => left.username.localeCompare(right.username),
        ),
      );
      setNewUsername("");
      setNewIsAdmin(false);
      toast.success(msg("settings.admin.accounts.created"));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyUsername(null);
    }
  }, [newIsAdmin, newUsername]);

  const setRole = React.useCallback(async (account: ManagedAccount, next: boolean) => {
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
  }, []);

  const removeAccount = React.useCallback(async (account: ManagedAccount) => {
    if (
      !window.confirm(msg("settings.admin.accounts.delete_confirm", { username: account.username }))
    ) {
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
  }, []);

  const updateRowBudget = React.useCallback(
    async (targetUsername: string, nextBytes: number) => {
      const before = accounts;
      setAccounts((prev) =>
        prev.map((row) =>
          row.username === targetUsername
            ? { ...row, quota_bytes: nextBytes, effective_bytes: nextBytes }
            : row,
        ),
      );
      try {
        const saved = await setStorageQuotaOverride(targetUsername, nextBytes);
        setAccounts((prev) =>
          prev.map((row) =>
            row.username === targetUsername
              ? {
                  ...row,
                  quota_bytes: saved.quota_bytes,
                  effective_bytes: saved.effective_bytes,
                  used_bytes: saved.used_bytes,
                  quota_updated_by: saved.updated_by,
                }
              : row,
          ),
        );
        toast.success(msg("settings.admin.storage.saved"));
      } catch (err) {
        setAccounts(before);
        throw err;
      }
    },
    [accounts],
  );

  const resetRowBudget = React.useCallback(async (targetUsername: string) => {
    setBusyUsername(targetUsername);
    try {
      const restored = await deleteStorageQuotaOverride(targetUsername);
      setAccounts((prev) =>
        prev.map((row) =>
          row.username === targetUsername
            ? {
                ...row,
                quota_bytes: null,
                effective_bytes: restored.effective_bytes,
                used_bytes: restored.used_bytes,
                quota_updated_by: null,
              }
            : row,
        ),
      );
      toast.success(msg("settings.admin.storage.deleted"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyUsername(null);
    }
  }, []);

  const trimmedNewUsername = newUsername.trim().toLocaleLowerCase();
  const triggerLabel =
    accounts.length === 0
      ? msg("settings.admin.storage.view_list")
      : `${msg("settings.admin.storage.view_list")} (${accounts.length})`;

  return (
    <section className="space-y-3 rounded-lg border border-border/50 p-3">
      <div className="flex items-center gap-2">
        <User className="size-4 text-muted-foreground" aria-hidden="true" />
        <div>
          <h3 className="text-sm font-semibold">{msg("settings.admin.accounts.title")}</h3>
          <p className="text-xs text-muted-foreground">
            {msg("settings.admin.accounts.description")}
          </p>
          {defaultBytes != null && (
            <p className="text-xs text-muted-foreground">
              {msg("settings.admin.storage.default_budget", {
                value: formatStorageSize(defaultBytes),
              })}
            </p>
          )}
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto] sm:items-center">
        <Input
          value={newUsername}
          onChange={(event) => setNewUsername(event.target.value)}
          placeholder={msg("settings.admin.accounts.username_placeholder")}
          autoComplete="off"
          dir="ltr"
        />
        <label className="flex min-h-10 items-center justify-between gap-2 text-xs sm:justify-start">
          {msg("settings.admin.accounts.admin")}
          <Switch checked={newIsAdmin} onCheckedChange={setNewIsAdmin} />
        </label>
        <Button
          onClick={() => void createAccount()}
          disabled={!newUsername.trim() || busyUsername !== null}
        >
          {busyUsername === trimmedNewUsername ? (
            <CircleNotch className="size-4 animate-spin" />
          ) : (
            <Plus className="size-4" />
          )}
          {msg("settings.admin.accounts.create")}
        </Button>
      </div>

      {!session?.backendAccessToken && (
        <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          {msg("settings.admin.storage.auth_missing")}
        </div>
      )}

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
              <User className="size-4 text-muted-foreground" aria-hidden="true" />
              <SheetTitle>{msg("settings.admin.accounts.title")}</SheetTitle>
            </div>
          </SheetHeader>

          <div className="flex items-center gap-3 border-b border-border/40 bg-muted/20 px-6 py-2">
            <span className="text-[0.6875rem] tabular-nums text-muted-foreground">
              {filteredAccounts.length === accounts.length
                ? accounts.length
                : `${filteredAccounts.length} / ${accounts.length}`}
            </span>
            {defaultBytes != null && (
              <span className="text-[0.6875rem] text-muted-foreground" dir="ltr">
                {msg("settings.admin.storage.default_budget", {
                  value: formatStorageSize(defaultBytes),
                })}
              </span>
            )}
            <ResetColumnsButton resize={colResize} />
            <ResetFiltersButton filters={colFilters} />
            <ExportTableMenu
              iconOnly
              align="end"
              className="ms-auto"
              disabled={loading || filteredAccounts.length === 0}
              getData={() => ({
                columns: ["username", "is_admin", "budget_bytes", "used_bytes", "updated_by"],
                rows: filteredAccounts.map((a) => ({
                  username: a.username,
                  is_admin: a.is_admin,
                  budget_bytes: a.effective_bytes,
                  used_bytes: a.used_bytes,
                  updated_by: a.quota_updated_by || msg("settings.admin.storage.default"),
                })),
                filename: "accounts",
              })}
            />
          </div>

          <div className="flex-1 overflow-auto">
            <div className="table-scroll">
              <Table style={{ minWidth: "640px" }}>
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
                      label={msg("settings.admin.accounts.admin")}
                      sortKey="is_admin"
                      currentSort={sortKey}
                      sortDir={sortDir}
                      onSort={toggleSort}
                      width={colResize.widths["is_admin"]}
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
                      sortKey="quota_updated_by"
                      currentSort={sortKey}
                      sortDir={sortDir}
                      onSort={toggleSort}
                      filterCol="quota_updated_by"
                      filterOptions={filterOptions.quota_updated_by}
                      filters={colFilters.filters}
                      onFilter={colFilters.setColumnFilter}
                      openFilter={colFilters.openFilter}
                      setOpenFilter={colFilters.setOpenFilter}
                      width={colResize.widths["quota_updated_by"]}
                      onResize={colResize.setColumnWidth}
                    />
                    <TableHead className="w-12" />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow className="border-border/40 bg-muted/10">
                    <TableCell className="text-center">
                      <Input
                        value={newUsername}
                        onChange={(event) => setNewUsername(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" && newUsername.trim() && busyUsername === null) {
                            event.preventDefault();
                            void createAccount();
                          }
                        }}
                        placeholder={msg("settings.admin.accounts.username_placeholder")}
                        autoComplete="off"
                        dir="ltr"
                        className="mx-auto h-8 max-w-[180px] text-xs"
                      />
                    </TableCell>
                    <TableCell className="text-center">
                      <Switch
                        checked={newIsAdmin}
                        onCheckedChange={setNewIsAdmin}
                        aria-label={msg("settings.admin.accounts.admin")}
                      />
                    </TableCell>
                    <TableCell className="text-center text-xs text-muted-foreground/60">—</TableCell>
                    <TableCell className="text-center text-xs text-muted-foreground/60">—</TableCell>
                    <TableCell className="text-center text-xs text-muted-foreground/60">—</TableCell>
                    <TableCell className="w-12 text-center">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            onClick={() => void createAccount()}
                            disabled={!newUsername.trim() || busyUsername !== null}
                            className="mx-auto size-8 text-muted-foreground hover:text-foreground"
                            aria-label={msg("settings.admin.accounts.create")}
                          >
                            {busyUsername !== null && busyUsername === trimmedNewUsername ? (
                              <CircleNotch className="size-3.5 animate-spin" />
                            ) : (
                              <Plus className="size-3.5" />
                            )}
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>{msg("settings.admin.accounts.create")}</TooltipContent>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                  {filteredAccounts.length === 0 ? (
                    <TableRow>
                      <TableCell
                        colSpan={6}
                        className="px-6 py-10 text-center text-sm text-muted-foreground"
                      >
                        {accounts.length === 0
                          ? msg("settings.admin.accounts.empty")
                          : msg("settings.admin.storage.no_results")}
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredAccounts.map((account) => (
                      <TableRow
                        key={account.username}
                        className="border-border/40 hover:bg-accent/30"
                      >
                        <TableCell
                          className="max-w-[200px] truncate text-center font-semibold text-xs text-foreground"
                          dir="ltr"
                          title={account.username}
                        >
                          {account.username}
                        </TableCell>
                        <TableCell className="text-center">
                          <Switch
                            checked={account.is_admin}
                            onCheckedChange={(next) => void setRole(account, next)}
                            disabled={busyUsername !== null}
                            aria-label={`${msg("settings.admin.accounts.admin")}: ${account.username}`}
                          />
                        </TableCell>
                        <TableCell className="text-center">
                          <span className="inline-flex items-center justify-center gap-0.5">
                            <EditableBudgetCell
                              bytes={account.effective_bytes}
                              onSave={(nextBytes) => updateRowBudget(account.username, nextBytes)}
                              disabled={busyUsername !== null}
                            />
                            {account.quota_bytes != null && (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="icon-sm"
                                    onClick={() => void resetRowBudget(account.username)}
                                    disabled={busyUsername !== null}
                                    className="size-6 text-muted-foreground"
                                    aria-label={`${msg("settings.admin.storage.reset_override")}: ${account.username}`}
                                  >
                                    <ArrowCounterClockwise className="size-3" />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  {msg("settings.admin.storage.reset_override")}
                                </TooltipContent>
                              </Tooltip>
                            )}
                          </span>
                        </TableCell>
                        <TableCell className="text-center">
                          <UsageMeter used={account.used_bytes} budget={account.effective_bytes} />
                        </TableCell>
                        <TableCell
                          className="max-w-[180px] truncate text-center text-xs text-muted-foreground"
                          dir="ltr"
                          title={account.quota_updated_by || msg("settings.admin.storage.default")}
                        >
                          {account.quota_updated_by || msg("settings.admin.storage.default")}
                        </TableCell>
                        <TableCell className="w-12 text-center">
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon-sm"
                                onClick={() => void removeAccount(account)}
                                disabled={busyUsername !== null}
                                className="mx-auto size-8 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                                aria-label={`${msg("settings.admin.accounts.delete")}: ${account.username}`}
                              >
                                {busyUsername === account.username ? (
                                  <CircleNotch className="size-3.5 animate-spin" />
                                ) : (
                                  <Trash className="size-3.5" />
                                )}
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                              {msg("settings.admin.accounts.delete")}
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
    </section>
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

      <SettingsRow icon={HardDrives} label={msg("settings.about.api_url.label")}>
        <span className="max-w-full break-all text-xs font-mono text-muted-foreground" dir="ltr">
          {apiUrl}
        </span>
      </SettingsRow>

      <SettingsRow
        icon={ArrowCounterClockwise}
        label={msg("settings.about.reset_all.label")}
        description={msg("settings.about.reset_all.description")}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              size="icon-sm"
              onClick={handleResetAll}
              className="size-[44px] text-destructive hover:text-destructive sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
              aria-label={msg("settings.about.reset_all.action")}
            >
              <ArrowCounterClockwise className="size-3.5" />
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

  const formatTimestamp = (iso: string) => new Date(iso).toLocaleString(getActiveIntlLocale());
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

      <SettingsRow icon={Key} label={msg("settings.api.title")}>
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
                  className="size-[44px] text-destructive hover:text-destructive sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
                  aria-label={msg("settings.api.revoke")}
                >
                  <Trash className="size-3.5" />
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
                  className="size-[44px] sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
                  aria-label={msg("settings.api.generate")}
                >
                  <Key className="size-3.5" />
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
                <CopyButton
                  text={revealed ?? ""}
                  ariaLabel={msg("settings.api.copy")}
                  copiedAriaLabel={msg("settings.api.copied")}
                  variant="outline"
                  className="shrink-0"
                />
              </TooltipTrigger>
              <TooltipContent>{msg("settings.api.copy")}</TooltipContent>
            </Tooltip>
          </div>
          <Button variant="outline" size="sm" onClick={() => setRevealed(null)} className="w-full">
            {msg("settings.api.done")}
          </Button>
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
              className="size-[44px] sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
              aria-label={msg("settings.api.docs_action")}
            >
              <a href={docsUrl} target="_blank" rel="noopener noreferrer">
                <ArrowSquareOut className="size-3.5" />
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
  "tagging",
  "agent",
  "account",
  "providers",
  "api",
  "admin",
  "about",
] as const;
type SettingsTab = (typeof SETTINGS_TAB_ORDER)[number];
type SettingsMessageKey = Parameters<typeof msg>[0];

const SETTINGS_TAB_META: Record<
  SettingsTab,
  {
    icon: Icon;
    labelKey: SettingsMessageKey;
    group: "workflows" | "assistants" | "preferences" | "access" | "system";
  }
> = {
  wizard: {
    icon: Sparkle,
    labelKey: "settings.tab.wizard",
    group: "workflows",
  },
  tagging: {
    icon: Tag,
    labelKey: "settings.tab.tagging",
    group: "workflows",
  },
  agent: {
    icon: Robot,
    labelKey: "settings.tab.agent",
    group: "assistants",
  },
  account: {
    icon: User,
    labelKey: "settings.tab.account",
    group: "preferences",
  },
  providers: {
    icon: Plug,
    labelKey: "settings.tab.providers",
    group: "access",
  },
  api: {
    icon: Key,
    labelKey: "settings.tab.api",
    group: "access",
  },
  admin: {
    icon: HardDrive,
    labelKey: "settings.tab.admin",
    group: "system",
  },
  about: {
    icon: Info,
    labelKey: "settings.tab.about",
    group: "system",
  },
};

const SETTINGS_GROUPS = [
  { key: "workflows", labelKey: "settings.group.workflows" },
  { key: "assistants", labelKey: "settings.group.assistants" },
  { key: "preferences", labelKey: "settings.group.preferences" },
  { key: "access", labelKey: "settings.group.access" },
  { key: "system", labelKey: "settings.group.system" },
] as const;

// Vertical-rail item, styled to match the main sidebar nav while staying easy
// to scan when the list is filtered. On mobile the rail becomes a horizontal
// strip, so each item drops back to its intrinsic width.
const SETTINGS_RAIL_ITEM_CLASS =
  "min-h-[44px] w-full flex-none justify-start gap-2.5 rounded-lg px-3 py-2 font-medium text-sidebar-foreground/60 data-[state=inactive]:hover:bg-sidebar-accent/40 data-[state=inactive]:hover:text-sidebar-foreground data-[state=active]:bg-transparent data-[state=active]:border-transparent data-[state=active]:font-medium data-[state=active]:text-primary data-[state=active]:hover:text-primary max-md:w-auto! md:min-h-0 [@media(hover:none)_and_(pointer:coarse)]:min-h-[44px]";

function SettingsPanelHeader({ tab }: { tab: SettingsTab }) {
  const { icon: Icon, labelKey } = SETTINGS_TAB_META[tab];
  return (
    <div className="mb-4 flex items-center gap-3 border-b border-border/50 pb-3">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/[0.08] text-primary">
        <Icon className="size-4" aria-hidden="true" />
      </span>
      <h2 className="text-base font-semibold tracking-tight text-foreground">{msg(labelKey)}</h2>
    </div>
  );
}

export function SettingsModal() {
  const { open, setOpen, targetTab, clearTarget } = useSettingsModal();
  const { state: tutorialState } = useTutorialContext();
  const { data: session } = useSession();
  const isAdmin = session?.user?.role === "admin";
  const isPhone = useIsPhone();
  const prefersReduced = useReducedMotion();
  const [activeTab, setActiveTab] = React.useState<SettingsTab>(isPhone ? "account" : "wizard");
  const selectTab = React.useCallback((tab: SettingsTab) => {
    setActiveTab(tab);
    track(TelemetryEvent.SettingsTabChanged, { tab });
  }, []);
  const tabs = React.useMemo(
    () =>
      SETTINGS_TAB_ORDER.filter(
        (tab) => (isAdmin || tab !== "admin") && (!isPhone || isPhoneSettingsTab(tab)),
      ),
    [isAdmin, isPhone],
  );
  React.useEffect(() => {
    if (!tabs.includes(activeTab)) setActiveTab(isPhone ? "account" : "wizard");
  }, [activeTab, tabs, isPhone]);
  // Honor a deep-link (for example, a model picker opening Providers): when something opens the
  // modal targeting a tab, jump there once, then clear so a later manual open
  // keeps whatever tab the user last left it on.
  React.useEffect(() => {
    if (!targetTab) return;
    if ((tabs as readonly string[]).includes(targetTab)) {
      setActiveTab(targetTab as SettingsTab);
    }
    clearTarget();
  }, [targetTab, tabs, clearTarget]);
  // Fire settings_opened on the closed→open transition only; the ref stops a
  // re-render with open still true from re-emitting it.
  const wasOpen = React.useRef(false);
  React.useEffect(() => {
    if (open && !wasOpen.current) track(TelemetryEvent.SettingsOpened);
    wasOpen.current = open;
  }, [open]);
  return (
    <Dialog open={open} onOpenChange={setOpen} modal={!tutorialState.isVisible}>
      <DialogContent
        data-settings-text-buttons
        className="max-h-[calc(100dvh-1rem)] w-[calc(100vw-1rem)] gap-0 overflow-hidden p-0 sm:max-w-4xl [&_[data-slot=button]]:min-h-[44px] [&_[data-slot=button]]:min-w-[44px] [&_[data-slot=select-trigger]]:min-h-[44px] sm:[&_[data-slot=button]]:min-h-0 sm:[&_[data-slot=button]]:min-w-0 sm:[&_[data-slot=select-trigger]]:min-h-0 [@media(hover:none)_and_(pointer:coarse)]:[&_[data-slot=button]]:min-h-[44px] [@media(hover:none)_and_(pointer:coarse)]:[&_[data-slot=button]]:min-w-[44px] [@media(hover:none)_and_(pointer:coarse)]:[&_[data-slot=select-trigger]]:min-h-[44px]"
      >
        <DialogHeader className="border-b border-border/40 px-4 py-3 pe-12 text-start sm:px-5 sm:py-4">
          <div className="min-w-0">
            <DialogTitle>{msg("settings.title")}</DialogTitle>
            <DialogDescription className="mt-1 text-xs">
              {msg("settings.subtitle")}
            </DialogDescription>
          </div>
        </DialogHeader>

        <Tabs
          orientation="vertical"
          value={activeTab}
          onValueChange={(v) => selectTab(v as SettingsTab)}
          className="flex h-[calc(100dvh-5.75rem)] max-h-[680px] min-h-0 flex-col gap-0 sm:h-[min(72vh,680px)] md:flex-row"
        >
          <TabsList
            aria-label={msg("settings.title")}
            data-tutorial="settings-navigation"
            className="relative flex h-auto w-full shrink-0 items-stretch justify-start gap-1 overflow-x-auto rounded-none border-0 border-b border-border/40 bg-transparent px-3 pb-3 pt-2 shadow-none no-scrollbar max-md:flex-row! md:w-[220px] md:overflow-x-visible md:overflow-y-auto md:border-b-0 md:border-e"
          >
            {SETTINGS_GROUPS.map((group) => (
              <div key={group.key} className="contents md:block">
                <p className="hidden px-3 pb-1 pt-3 text-[0.625rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground/60 first:pt-1 md:block">
                  {msg(group.labelKey)}
                </p>
                {tabs
                  .filter((tab) => SETTINGS_TAB_META[tab].group === group.key)
                  .map((tab) => {
                    const { icon: Icon, labelKey } = SETTINGS_TAB_META[tab];
                    return (
                      <TabsTrigger key={tab} value={tab} className={SETTINGS_RAIL_ITEM_CLASS}>
                        {tab === activeTab && (
                          <motion.div
                            layoutId="settings-rail-active"
                            className="absolute inset-0 rounded-lg bg-primary/[0.08] ring-1 ring-primary/10"
                            transition={
                              prefersReduced
                                ? { duration: 0 }
                                : { type: "spring", stiffness: 350, damping: 28 }
                            }
                          />
                        )}
                        <span className="relative z-10 flex min-w-0 flex-1 items-center gap-2.5">
                          <Icon
                            aria-hidden="true"
                            className="size-4 shrink-0 transition-colors duration-200"
                          />
                          <span className="flex-1 truncate">{msg(labelKey)}</span>
                        </span>
                      </TabsTrigger>
                    );
                  })}
              </div>
            ))}
          </TabsList>

          <div className="min-w-0 flex-1 overscroll-contain overflow-y-auto px-4 py-4 md:px-6 md:py-5">
            <motion.div
              key={activeTab}
              initial={prefersReduced ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: prefersReduced ? 0 : 0.18, ease: [0.2, 0.8, 0.2, 1] }}
            >
              <SettingsPanelHeader tab={activeTab} />
              <TabsContent value="wizard">
                <WizardTab />
              </TabsContent>
              <TabsContent value="tagging">
                <TaggingTab />
              </TabsContent>
              <TabsContent value="agent">
                <AgentTab />
              </TabsContent>
              <TabsContent value="account" data-tutorial="settings-account">
                <AccountTab />
              </TabsContent>
              <TabsContent value="providers" data-tutorial="settings-providers">
                <ByokKeysSection />
              </TabsContent>
              <TabsContent value="api">
                <ApiTab />
              </TabsContent>
              {isAdmin && (
                <TabsContent value="admin">
                  <AdminTab />
                </TabsContent>
              )}
              <TabsContent value="about">
                <AboutTab />
              </TabsContent>
            </motion.div>
          </div>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
