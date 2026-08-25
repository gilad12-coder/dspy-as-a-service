"use client";

import * as React from "react";
import { createPortal } from "react-dom";
import { useSession } from "next-auth/react";
import { toast } from "react-toastify";
import { motion, useReducedMotion } from "framer-motion";
import {
  ChartBar,
  ChatText,
  BookOpen,
  Robot,
  Brain,
  Columns,
  Cpu,
  CreditCard,
  Key,
  ArrowSquareOut,
  Feather,
  HardDrive,
  Keyboard,
  Lock,
  type Icon,
  Translate,
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
import { WalletTab, UsageTab, ByokKeysSection } from "@/features/billing";
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
import { ComposerModelMenu } from "@/shared/ui/agent";
import { ModelChip } from "@/shared/ui/model-chip";
import { ModelConfigModal, useRecentModelConfigs } from "@/features/submit";
import { getActiveDir, getActiveIntlLocale } from "@/shared/lib/runtime-locale";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { LEGAL_CONFIG } from "@/features/legal";
import { LanguageSwitcher } from "@/shared/ui/language-switcher";
import { useTutorialContext } from "@/features/tutorial";
import {
  deleteStorageQuotaOverride,
  generateApiToken,
  getApiToken,
  getMemorySettings,
  getStorageQuotaOverrides,
  revokeApiToken,
  searchAdminUsers,
  setStorageQuotaOverride,
  updateMemorySettings,
  type ApiTokenInfo,
  type DirectoryUserMatch,
  type MemoryKnob,
  type MemoryKnobName,
  type MemorySettings,
  type StorageQuotaOverride,
} from "@/shared/lib/api";

import { useUserPrefs } from "../hooks/use-user-prefs";
import { useSettingsModal } from "../hooks/use-settings-modal";
import { useIsPhone } from "@/shared/hooks/use-device-class";
import { isPhoneSettingsTab } from "@/shared/lib/device-class";
import { ShortcutRecorder } from "./ShortcutRecorder";
import { PrivacyTab } from "./PrivacyTab";
import { SecurityTab } from "./SecurityTab";
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
        catalogModels={catalogModels ?? undefined}
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
      <SettingsRow
        icon={Cpu}
        label={msg("settings.agent.default_model.label")}
        description={msg("settings.agent.default_model.description")}
      >
        <ComposerModelMenu
          value={prefs.composerModel}
          onChange={(v) => setPref("composerModel", v)}
          effort={prefs.composerEffort}
          onEffortChange={(v) => setPref("composerEffort", v)}
        />
      </SettingsRow>

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

      <SettingsRow icon={Translate} label={msg("shared.language.switch_aria")}>
        <LanguageSwitcher />
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
  const [pos, setPos] = React.useState<{
    top: number;
    left: number;
    width: number;
    maxH: number;
  } | null>(null);
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
      const vals = [...new Set(overrides.map((o) => String(o[key] ?? "")).filter(Boolean))].sort();
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
      const saved = await setStorageQuotaOverride(
        normalizedUsername,
        pendingBudgetMb * BYTES_PER_MB,
      );
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
            ? msg("settings.admin.storage.default_budget", {
                value: formatStorageSize(defaultBytes),
              })
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
              disabled={loading || filteredOverrides.length === 0}
              getData={() => ({
                columns: ["username", "budget_bytes", "used_bytes", "updated_by"],
                rows: filteredOverrides.map((o) => ({
                  username: o.username,
                  budget_bytes: o.effective_bytes,
                  used_bytes: o.used_bytes,
                  updated_by: o.updated_by || msg("settings.admin.storage.default"),
                })),
                filename: "storage-quota-overrides",
              })}
            />
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
                    <TableCell className="text-center text-xs text-muted-foreground/70">
                      —
                    </TableCell>
                    <TableCell className="text-center text-xs text-muted-foreground/70">
                      —
                    </TableCell>
                    <TableCell className="w-12 text-center">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            onClick={() => void addPendingUser()}
                            disabled={busy || !pendingUsername.trim() || pendingBudgetMb === ""}
                            className="size-[44px] sm:size-8 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
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
                      <TableRow key={item.username} className="border-border/40 hover:bg-accent/30">
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
                            <TooltipContent>{msg("settings.admin.storage.delete")}</TooltipContent>
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
  const feedbackHref = `mailto:${LEGAL_CONFIG.contactEmail}?subject=${encodeURIComponent(
    msg("settings.about.feedback.subject"),
  )}`;

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
        icon={ChatText}
        label={msg("settings.about.feedback.label")}
        description={msg("settings.about.feedback.description")}
      >
        <Button variant="outline" size="sm" asChild>
          <a href={feedbackHref}>{msg("settings.about.feedback.action")}</a>
        </Button>
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
  "security",
  "privacy",
  "billing",
  "usage",
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
  security: {
    icon: ShieldCheck,
    labelKey: "settings.tab.security",
    group: "access",
  },
  privacy: {
    icon: Lock,
    labelKey: "settings.tab.privacy",
    group: "access",
  },
  billing: {
    icon: CreditCard,
    labelKey: "settings.tab.billing",
    group: "access",
  },
  usage: {
    icon: ChartBar,
    labelKey: "settings.tab.usage",
    group: "access",
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
  // Honor a deep-link (e.g. the credit chip → wallet): when something opens the
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
              <TabsContent value="security">
                <SecurityTab />
              </TabsContent>
              <TabsContent value="privacy" data-tutorial="settings-privacy">
                <PrivacyTab />
              </TabsContent>
              <TabsContent value="billing">
                <WalletTab />
              </TabsContent>
              <TabsContent value="usage">
                <UsageTab />
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
