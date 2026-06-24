"use client";

import * as React from "react";
import {
  Check,
  History,
  MoreHorizontal,
  Pencil,
  Pin,
  PinOff,
  Search,
  Trash2,
} from "lucide-react";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/shared/ui/primitives/sheet";
import { Input } from "@/shared/ui/primitives/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/shared/ui/primitives/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { cn } from "@/shared/lib/utils";
import { msg } from "@/shared/lib/messages";
import { ConversationDrawerSkeleton } from "./ConversationDrawerSkeleton";

import type { ConversationSummary } from "../lib/conversation-api";

interface ConversationDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  conversations: ConversationSummary[];
  loading: boolean;
  activeId: string | null;
  unreadIds: ReadonlySet<string>;
  query: string;
  onQueryChange: (q: string) => void;
  onPick: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onTogglePin: (id: string, pinned: boolean) => void;
  onDelete: (id: string) => void;
}

// Bucket conversations into pinned → concrete calendar dates, mirroring the
// optimizations sidebar grouping (features/sidebar/lib/group-jobs.ts) so the
// two histories use the same visual rhythm. Date labels are formatted in
// he-IL DD/MM/YYYY.
interface ConversationGroup {
  label: string;
  rows: ConversationSummary[];
}

const DATE_FORMATTER = new Intl.DateTimeFormat("he-IL", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
});

function groupConversationsByRecency(rows: ConversationSummary[]): ConversationGroup[] {
  const pinned: ConversationSummary[] = [];
  const dated = new Map<string, ConversationGroup>();

  for (const row of rows) {
    if (row.pinned) {
      pinned.push(row);
      continue;
    }
    const updated = new Date(row.updatedAt ?? 0);
    const validDate = !Number.isNaN(updated.getTime());
    const key = validDate
      ? `${updated.getFullYear()}-${String(updated.getMonth() + 1).padStart(2, "0")}-${String(
          updated.getDate(),
        ).padStart(2, "0")}`
      : "unknown";
    const label = validDate
      ? DATE_FORMATTER.format(updated)
      : msg("auto.features.agent.panel.components.conversationdrawer.section_unknown_date");
    const group = dated.get(key) ?? { label, rows: [] };
    group.rows.push(row);
    dated.set(key, group);
  }

  const groups: ConversationGroup[] = [];
  if (pinned.length > 0) {
    groups.push({
      label: msg("auto.features.agent.panel.components.conversationdrawer.section_pinned"),
      rows: pinned,
    });
  }
  groups.push(...dated.values());
  return groups;
}

export function ConversationDrawer(props: ConversationDrawerProps) {
  const {
    open,
    onOpenChange,
    conversations,
    loading,
    activeId,
    unreadIds,
    query,
    onQueryChange,
    onPick,
    onRename,
    onTogglePin,
    onDelete,
  } = props;
  const groups = groupConversationsByRecency(conversations);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="left"
        className="w-[min(420px,90vw)] sm:max-w-none p-0 flex flex-col"
      >
        <SheetHeader className="border-b border-border/40 p-3">
          <SheetTitle className="text-[0.875rem] flex items-center gap-2">
            <History className="size-4" aria-hidden="true" />
            {msg("auto.features.agent.panel.components.conversationdrawer.title")}
          </SheetTitle>
          <div className="relative mt-2">
            <Search
              className="absolute top-1/2 -translate-y-1/2 end-2 size-3.5 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              placeholder={msg(
                "auto.features.agent.panel.components.conversationdrawer.search_placeholder",
              )}
              className="h-8 text-[0.8125rem] pe-7"
            />
          </div>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto px-2 pb-4">
          {loading && conversations.length === 0 ? (
            <ConversationDrawerSkeleton />
          ) : conversations.length === 0 ? (
            <p className="px-3 py-6 text-center text-[0.75rem] text-muted-foreground">
              {query.trim()
                ? msg("auto.features.agent.panel.components.conversationdrawer.no_results")
                : msg("auto.features.agent.panel.components.conversationdrawer.empty")}
            </p>
          ) : (
            <>
              {groups.map((group) => (
                <Section
                  key={group.label}
                  label={group.label}
                  rows={group.rows}
                  activeId={activeId}
                  unreadIds={unreadIds}
                  onPick={onPick}
                  onRename={onRename}
                  onTogglePin={onTogglePin}
                  onDelete={onDelete}
                />
              ))}
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

interface SectionProps {
  label: string;
  rows: ConversationSummary[];
  activeId: string | null;
  unreadIds: ReadonlySet<string>;
  onPick: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onTogglePin: (id: string, pinned: boolean) => void;
  onDelete: (id: string) => void;
}

function Section({
  label,
  rows,
  activeId,
  unreadIds,
  onPick,
  onRename,
  onTogglePin,
  onDelete,
}: SectionProps) {
  if (rows.length === 0) return null;
  return (
    <div className="mt-3">
      <div className="px-2 pb-1 text-[0.6875rem] uppercase tracking-wide text-muted-foreground/80">
        {label}
      </div>
      <ul className="space-y-0.5">
        {rows.map((row) => (
          <ConversationRow
            key={row.id}
            row={row}
            active={row.id === activeId}
            unread={unreadIds.has(row.id)}
            onPick={onPick}
            onRename={onRename}
            onTogglePin={onTogglePin}
            onDelete={onDelete}
          />
        ))}
      </ul>
    </div>
  );
}

interface RowProps {
  row: ConversationSummary;
  active: boolean;
  unread: boolean;
  onPick: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onTogglePin: (id: string, pinned: boolean) => void;
  onDelete: (id: string) => void;
}

function ConversationRow({ row, active, unread, onPick, onRename, onTogglePin, onDelete }: RowProps) {
  const [editing, setEditing] = React.useState(false);
  const [draft, setDraft] = React.useState(row.title);
  const [menuOpen, setMenuOpen] = React.useState(false);

  React.useEffect(() => {
    if (!editing) setDraft(row.title);
  }, [editing, row.title]);

  const commit = React.useCallback(() => {
    const trimmed = draft.trim();
    setEditing(false);
    if (!trimmed || trimmed === row.title) return;
    onRename(row.id, trimmed);
  }, [draft, onRename, row.id, row.title]);

  if (editing) {
    return (
      <li>
        <div className="px-2 py-1.5">
          <Input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                commit();
              } else if (e.key === "Escape") {
                e.preventDefault();
                setEditing(false);
              }
            }}
            className="h-7 text-[0.8125rem]"
          />
        </div>
      </li>
    );
  }

  return (
    <li>
      <div
        className={cn(
          "group flex items-center gap-1.5 rounded-md px-2 py-1.5 cursor-pointer",
          active ? "bg-accent" : "hover:bg-accent/60",
        )}
        onClick={() => onPick(row.id)}
      >
        {unread && !active && (
          <span
            aria-label={msg(
              "auto.features.agent.panel.components.conversationdrawer.unread_indicator",
            )}
            className="size-1.5 rounded-full bg-primary shrink-0"
          />
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1 truncate text-[0.8125rem]">
            {row.pinned && <Pin className="size-3 text-muted-foreground shrink-0" />}
            <span
              className={cn(
                "truncate",
                unread && !active && "font-semibold text-foreground",
              )}
            >
              {row.title ||
                msg("auto.features.agent.panel.components.conversationdrawer.untitled")}
            </span>
          </div>
          {row.preview && (
            <div className="truncate text-[0.6875rem] text-muted-foreground">{row.preview}</div>
          )}
        </div>
        <Popover open={menuOpen} onOpenChange={setMenuOpen}>
          <Tooltip>
            <TooltipTrigger asChild>
              <PopoverTrigger asChild>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                  }}
                  className={cn(
                    "rounded-md p-1 text-muted-foreground hover:bg-accent/70 hover:text-foreground transition-colors cursor-pointer",
                    "opacity-0 group-hover:opacity-100 data-[state=open]:opacity-100",
                  )}
                  aria-label={msg(
                    "auto.features.agent.panel.components.conversationdrawer.row_menu",
                  )}
                >
                  <MoreHorizontal className="size-3.5" />
                </button>
              </PopoverTrigger>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              {msg("auto.features.agent.panel.components.conversationdrawer.row_menu")}
            </TooltipContent>
          </Tooltip>
          <PopoverContent
            side="bottom"
            align="end"
            sideOffset={4}
            className="w-48 p-1"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => {
                setEditing(true);
                setMenuOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-[0.8125rem] hover:bg-accent cursor-pointer"
            >
              <Pencil className="size-3.5" />
              {msg("auto.features.agent.panel.components.conversationdrawer.rename")}
            </button>
            <button
              type="button"
              onClick={() => {
                onTogglePin(row.id, !row.pinned);
                setMenuOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-[0.8125rem] hover:bg-accent cursor-pointer"
            >
              {row.pinned ? <PinOff className="size-3.5" /> : <Pin className="size-3.5" />}
              {row.pinned
                ? msg("auto.features.agent.panel.components.conversationdrawer.unpin")
                : msg("auto.features.agent.panel.components.conversationdrawer.pin")}
            </button>
            <button
              type="button"
              onClick={() => {
                onDelete(row.id);
                setMenuOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-[0.8125rem] text-destructive hover:bg-destructive/10 cursor-pointer"
            >
              <Trash2 className="size-3.5" />
              {msg("auto.features.agent.panel.components.conversationdrawer.delete")}
            </button>
            {active && (
              <div className="border-t border-border/60 mt-1 pt-1 px-2 py-1 text-[0.6875rem] text-muted-foreground flex items-center gap-1">
                <Check className="size-3" />
                {msg("auto.features.agent.panel.components.conversationdrawer.active_hint")}
              </div>
            )}
          </PopoverContent>
        </Popover>
      </div>
    </li>
  );
}

