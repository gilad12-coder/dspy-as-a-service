"use client";

import * as React from "react";
import Link from "next/link";
import {
  Check,
  Database,
  FileText,
  ChatText,
  Sparkle,
  Trash,
  ArrowUpLeft,
  ArrowUpRight,
  type Icon,
} from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { cn } from "@/shared/lib/utils";
import { formatStorageSize } from "@/shared/lib/formatters";
import { msg, type MessageKey } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import type { StorageItem } from "@/shared/lib/api";

/** Per-type icon, in-app jump target, and label key for a ranked storage item. */
const TYPE_META: Record<
  StorageItem["type"],
  { icon: Icon; label: MessageKey; href: (id: string) => string | null }
> = {
  optimization: {
    icon: Sparkle,
    label: "storage.type.optimization",
    href: (id) => `/optimizations/${id}`,
  },
  dataset: {
    icon: Database,
    label: "storage.type.dataset",
    href: (id) => `/datasets?open=${id}`,
  },
  // Chats live in the slide-over agent panel; ?chat=<id> opens the panel onto
  // that conversation from whichever route the link is clicked on.
  chat: {
    icon: ChatText,
    label: "storage.type.chat",
    href: (id) => `?chat=${encodeURIComponent(id)}`,
  },
  // Pending uploads are transient wizard rows with no page of their own.
  staged_upload: {
    icon: FileText,
    label: "storage.type.staged_upload",
    href: () => null,
  },
};

/** Inputs for one row in a storage cleanup list. */
interface StorageItemRowProps {
  item: StorageItem;
  /** Whether this row is in the multi-select set. */
  selected: boolean;
  /** Toggle this row's selection; ``shiftKey`` requests a contiguous range from
   *  the previously toggled row. The row body is otherwise inert. */
  onToggle: (item: StorageItem, shiftKey: boolean) => void;
  onDelete: (item: StorageItem) => void;
  /** Fired when the open link is followed, so a host drawer can close itself
   *  instead of staying in front of the destination it just navigated to. */
  onNavigate?: () => void;
}

/**
 * One ranked item: a leading select checkbox, a type glyph, the item's name and
 * kind, its size, an optional jump to where it lives, and an in-place delete.
 * Only the checkbox (and the open/delete controls) is interactive — the row body
 * is inert so selection is never triggered by a stray click.
 */
export function StorageItemRow({
  item,
  selected,
  onToggle,
  onDelete,
  onNavigate,
}: StorageItemRowProps) {
  const meta = TYPE_META[item.type];
  const Icon = meta.icon;
  const href = meta.href(item.id);
  // The jump-to glyph points toward the reading-forward direction: up-left in
  // RTL (Hebrew), mirrored to up-right in LTR (English).
  const OpenArrow = getActiveDir() === "rtl" ? ArrowUpLeft : ArrowUpRight;

  return (
    <li
      className={cn(
        "group flex items-center gap-3 rounded-lg px-2 py-2.5 transition-colors duration-150",
        selected ? "bg-[#C8A882]/12" : "hover:bg-muted/40",
      )}
    >
      <button
        type="button"
        role="checkbox"
        aria-checked={selected}
        aria-label={msg("storage.select.item")}
        onClick={(event) => onToggle(item, event.shiftKey)}
        className={cn(
          "grid size-[44px] shrink-0 cursor-pointer place-items-center rounded-md border transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 lg:size-5",
          selected
            ? "border-transparent bg-foreground text-background"
            : "border-border/70 bg-background hover:border-foreground/40",
        )}
      >
        {selected && <Check className="size-3.5" aria-hidden="true" />}
      </button>
      <div className="grid size-9 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground">
        <Icon className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-foreground">
          <bdi>{item.name}</bdi>
        </p>
        <p className="text-xs text-muted-foreground">{msg(meta.label)}</p>
        <p className="text-xs tabular-nums text-muted-foreground sm:hidden">
          {formatStorageSize(item.bytes)}
        </p>
      </div>
      <span className="hidden shrink-0 text-sm tabular-nums text-muted-foreground sm:inline">
        {formatStorageSize(item.bytes)}
      </span>
      <div className="flex shrink-0 items-center gap-0.5">
        {href && (
          <Button
            asChild
            variant="ghost"
            size="icon-sm"
            aria-label={msg("storage.item.open")}
            className="!size-[44px] lg:!size-8"
          >
            <Link href={href} onClick={() => onNavigate?.()}>
              <OpenArrow className="size-4" />
            </Link>
          </Button>
        )}
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => onDelete(item)}
          aria-label={msg("storage.item.delete")}
          className="size-[44px] text-muted-foreground hover:bg-destructive/10 hover:text-destructive lg:size-8"
        >
          <Trash className="size-4" />
        </Button>
      </div>
    </li>
  );
}
