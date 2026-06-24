"use client";

import * as React from "react";
import {
  Calendar,
  Check,
  Component,
  Cpu,
  Layers,
  Search,
  Target,
  X,
} from "lucide-react";
import { msg, formatMsg } from "@/shared/lib/messages";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/shared/ui/primitives/sheet";
import { SkynetDatePicker } from "./SkynetDatePicker";

interface FiltersDrawerProps {
  open: boolean;
  onOpenChange: (next: boolean) => void;
  /** All distinct model identifiers in the corpus (sorted alphabetically by caller). */
  modelOptions: string[];
  /** All distinct optimizer names in the corpus (sorted alphabetically by caller). */
  optimizerOptions: string[];
  /** All distinct module names in the corpus (sorted alphabetically by caller). */
  moduleOptions: string[];
  /** Currently active filter values. */
  selectedModels: string[];
  selectedOptimizers: string[];
  selectedTypes: string[];
  selectedModules: string[];
  dateFrom: string | null;
  dateTo: string | null;
  onChangeModels: (next: string[]) => void;
  onChangeOptimizers: (next: string[]) => void;
  onChangeTypes: (next: string[]) => void;
  onChangeModules: (next: string[]) => void;
  onChangeDateRange: (from: string | null, to: string | null) => void;
  /** Wipes every filter inside the drawer (excluding free-text query). */
  onClearAll: () => void;
}

const TYPE_VALUES: ReadonlyArray<{
  value: string;
  labelKey: Parameters<typeof msg>[0];
}> = [
  { value: "run", labelKey: "explore.filter.run" },
  { value: "grid_search", labelKey: "explore.filter.grid" },
];

// Lists longer than this get an inline search box. Shorter lists already
// fit in a few rows of chips — adding a search input there is friction.
const SEARCH_THRESHOLD = 8;

/**
 * Slide-in panel for structured filtering on top of the free-text query.
 * Sections stack vertically and never collapse — discoverability beats
 * compactness here, since most users will scan once and pick a handful.
 * Long option lists (models, optimizers) get a per-section search input;
 * date inputs use the project's custom calendar to stay consistent with
 * the rest of the visual system.
 */
export function FiltersDrawer({
  open,
  onOpenChange,
  modelOptions,
  optimizerOptions,
  moduleOptions,
  selectedModels,
  selectedOptimizers,
  selectedTypes,
  selectedModules,
  dateFrom,
  dateTo,
  onChangeModels,
  onChangeOptimizers,
  onChangeTypes,
  onChangeModules,
  onChangeDateRange,
  onClearAll,
}: FiltersDrawerProps) {
  const totalActive =
    selectedModels.length +
    selectedOptimizers.length +
    selectedTypes.length +
    selectedModules.length +
    (dateFrom ? 1 : 0) +
    (dateTo ? 1 : 0);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        showCloseButton={false}
        className="w-full !max-w-md gap-0 border-border bg-background p-0"
      >
        <div className="flex h-full flex-col">
          <SheetHeader className="flex-row items-start justify-between gap-3 border-b border-border/60 px-6 py-5">
            <div className="flex flex-col gap-1.5">
              <SheetTitle className="text-[17px] font-medium tracking-tight text-foreground">
                {msg("explore.filters.title")}
              </SheetTitle>
              <SheetDescription className="text-[12.5px] leading-relaxed text-foreground/55">
                {msg("explore.filters.subtitle")}
              </SheetDescription>
            </div>
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              aria-label={msg("explore.filters.close")}
              className="inline-flex size-9 shrink-0 items-center justify-center rounded-lg text-foreground/55 transition-[background-color,color] cursor-pointer hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              <X className="size-4" aria-hidden="true" />
            </button>
          </SheetHeader>

          <div className="flex-1 overflow-y-auto px-6 py-6">
            {/* Two clusters, each held tighter (gap-7) than the space between
                them (gap-10): the optimization itself (program, model,
                optimizer), then the run's own metadata (kind, date). */}
            <div className="flex flex-col gap-10">
              <div className="flex flex-col gap-7">
                <SearchableChipSection
                  title={msg("explore.filters.section.modules")}
                  icon={Component}
                  options={moduleOptions}
                  selected={selectedModules}
                  onToggle={(v) =>
                    onChangeModules(toggleValue(selectedModules, v))
                  }
                  dir="auto"
                />

                <SearchableChipSection
                  title={msg("explore.filters.section.models")}
                  icon={Cpu}
                  options={modelOptions}
                  selected={selectedModels}
                  onToggle={(v) => onChangeModels(toggleValue(selectedModels, v))}
                  dir="ltr"
                />

                <SearchableChipSection
                  title={msg("explore.filters.section.optimizers")}
                  icon={Target}
                  options={optimizerOptions}
                  selected={selectedOptimizers}
                  onToggle={(v) =>
                    onChangeOptimizers(toggleValue(selectedOptimizers, v))
                  }
                  dir="ltr"
                />
              </div>

              <div className="flex flex-col gap-7">
                <FilterSection
                  title={msg("explore.filters.section.types")}
                  icon={Layers}
                  selectedCount={selectedTypes.length}
                >
                  <ChipGroup
                    options={TYPE_VALUES.map((t) => t.value)}
                    labels={Object.fromEntries(
                      TYPE_VALUES.map((t) => [t.value, msg(t.labelKey)]),
                    )}
                    selected={selectedTypes}
                    onToggle={(v) =>
                      onChangeTypes(toggleValue(selectedTypes, v))
                    }
                    dir="auto"
                  />
                </FilterSection>

                <FilterSection
                  title={msg("explore.filters.section.date")}
                  icon={Calendar}
                  selectedCount={(dateFrom ? 1 : 0) + (dateTo ? 1 : 0)}
                >
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <DateRangeField
                      label={msg("explore.filters.date.from")}
                      value={dateFrom}
                      max={dateTo ?? undefined}
                      onChange={(v) => onChangeDateRange(v, dateTo)}
                    />
                    <DateRangeField
                      label={msg("explore.filters.date.to")}
                      value={dateTo}
                      min={dateFrom ?? undefined}
                      onChange={(v) => onChangeDateRange(dateFrom, v)}
                    />
                  </div>
                </FilterSection>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between gap-3 border-t border-border/60 bg-background px-6 py-4">
            <button
              type="button"
              onClick={onClearAll}
              disabled={totalActive === 0}
              className="inline-flex items-center justify-center rounded-lg px-3 py-2 text-[13px] text-foreground/65 transition-[background-color,color] cursor-pointer hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:text-foreground/30 disabled:hover:bg-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              {msg("explore.filters.clear")}
            </button>
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              className="inline-flex items-center justify-center rounded-lg bg-foreground px-4 py-2 text-[13px] font-medium text-background transition-colors cursor-pointer hover:bg-foreground/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              {msg("explore.filters.apply")}
            </button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}

function toggleValue(current: string[], value: string): string[] {
  return current.includes(value)
    ? current.filter((v) => v !== value)
    : [...current, value];
}

type IconComponent = React.ComponentType<{
  className?: string;
  "aria-hidden"?: boolean | "true";
  strokeWidth?: number;
}>;

function FilterSection({
  title,
  icon: Icon,
  selectedCount = 0,
  children,
}: {
  title: string;
  icon?: IconComponent;
  selectedCount?: number;
  children: React.ReactNode;
}) {
  const active = selectedCount > 0;
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h3
          className={`inline-flex items-center gap-2 text-[12px] font-medium tracking-wide transition-colors ${
            active ? "text-foreground/80" : "text-foreground/55"
          }`}
        >
          {Icon && (
            <Icon
              className={`size-3.5 transition-colors ${
                active ? "text-foreground/70" : "text-foreground/45"
              }`}
              aria-hidden="true"
              strokeWidth={1.75}
            />
          )}
          <span>{title}</span>
        </h3>
        {active && (
          <span
            className="rounded-full bg-foreground/[0.06] px-1.5 py-0.5 text-[10.5px] font-medium tabular-nums text-foreground/60"
            aria-label={formatMsg(
              selectedCount === 1
                ? "explore.filters.section.selected"
                : "explore.filters.section.selected_many",
              { n: selectedCount },
            )}
          >
            {formatMsg(
              selectedCount === 1
                ? "explore.filters.section.selected"
                : "explore.filters.section.selected_many",
              { n: selectedCount },
            )}
          </span>
        )}
      </div>
      {children}
    </section>
  );
}

function SearchableChipSection({
  title,
  icon,
  options,
  selected,
  onToggle,
  dir,
}: {
  title: string;
  icon?: IconComponent;
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
  dir: "ltr" | "rtl" | "auto";
}) {
  const [query, setQuery] = React.useState("");
  const showSearch = options.length > SEARCH_THRESHOLD;
  const trimmed = query.trim().toLowerCase();

  // Selected values always render first, then the rest match the search.
  const visible = React.useMemo(() => {
    if (!trimmed) return options;
    return options.filter((v) => v.toLowerCase().includes(trimmed));
  }, [options, trimmed]);
  const ordered = React.useMemo(() => {
    const selectedSet = new Set(selected);
    const heads = visible.filter((v) => selectedSet.has(v));
    const tails = visible.filter((v) => !selectedSet.has(v));
    return [...heads, ...tails];
  }, [visible, selected]);

  return (
    <FilterSection title={title} icon={icon} selectedCount={selected.length}>
      {showSearch && (
        <SectionSearchInput
          value={query}
          onChange={setQuery}
          placeholder={formatMsg("explore.filters.section.search", {
            section: title,
          })}
        />
      )}
      {ordered.length === 0 ? (
        <p className="text-[12.5px] text-foreground/45">
          {trimmed
            ? msg("explore.filters.section.no_search_match")
            : msg("explore.filters.empty_section")}
        </p>
      ) : (
        <ChipGroup
          options={ordered}
          selected={selected}
          onToggle={onToggle}
          dir={dir}
        />
      )}
    </FilterSection>
  );
}

function SectionSearchInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (next: string) => void;
  placeholder: string;
}) {
  return (
    <div className="relative">
      <Search
        className="pointer-events-none absolute end-2.5 top-1/2 size-3.5 -translate-y-1/2 text-foreground/40"
        aria-hidden="true"
      />
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        dir="auto"
        className="w-full rounded-lg border border-border bg-background ps-3 pe-8 py-1.5 text-[12.5px] text-foreground placeholder:text-foreground/40 transition-colors hover:border-foreground/30 focus:border-foreground/40 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
      />
    </div>
  );
}

function ChipGroup({
  options,
  selected,
  onToggle,
  labels,
  dir,
}: {
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
  /** Optional override for display strings (e.g. type values → Hebrew). */
  labels?: Record<string, string>;
  /**
   * Per-chip text direction. LTR for code identifiers, RTL for Hebrew labels,
   * auto for user-authored names that may be either (tasks, modules).
   */
  dir: "ltr" | "rtl" | "auto";
}) {
  if (options.length === 0) {
    return (
      <p className="text-[12.5px] text-foreground/45">
        {msg("explore.filters.empty_section")}
      </p>
    );
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((value) => {
        const active = selected.includes(value);
        const label = labels?.[value] ?? value;
        return (
          <SelectableChip
            key={value}
            label={label}
            active={active}
            dir={dir}
            onClick={() => onToggle(value)}
          />
        );
      })}
    </div>
  );
}

function SelectableChip({
  label,
  active,
  dir,
  onClick,
}: {
  label: string;
  active: boolean;
  dir: "ltr" | "rtl" | "auto";
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      dir={dir}
      className={`group inline-flex max-w-full items-center gap-1 rounded-full border px-3 py-1.5 text-[12.5px] transition-[background-color,border-color,color] cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 ${
        active
          ? "border-foreground/40 bg-foreground/[0.06] text-foreground"
          : "border-border bg-background text-foreground/70 hover:border-foreground/30 hover:text-foreground"
      }`}
    >
      {active && (
        <Check
          className="size-3 shrink-0 text-foreground/70"
          aria-hidden="true"
          strokeWidth={2.25}
        />
      )}
      <span className="min-w-0 truncate tabular-nums">{label}</span>
    </button>
  );
}

function DateRangeField({
  label,
  value,
  onChange,
  min,
  max,
}: {
  label: string;
  value: string | null;
  onChange: (next: string | null) => void;
  min?: string;
  max?: string;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[11.5px] text-foreground/55">{label}</span>
      <SkynetDatePicker
        value={value}
        onChange={onChange}
        min={min ?? null}
        max={max ?? null}
        ariaLabel={label}
      />
    </label>
  );
}
