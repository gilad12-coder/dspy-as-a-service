"use client";

import * as React from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { motion } from "framer-motion";
import { MoreHorizontal, Pencil, Pin, Trash2 } from "lucide-react";
import { toast } from "react-toastify";

import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import { TAGGER_SESSIONS_CHANGED } from "@/features/tagger";
import {
  renameTaggerSession,
  setTaggerSessionPinned,
  type SidebarJobItem,
} from "@/shared/lib/api";

/**
 * One saved text-labeling session in the unified sidebar list. A trimmed
 * sibling of ``JobRow``: it links to ``/tagger/[id]`` to resume annotating and
 * offers only the actions that fit a labeling session — rename, pin, delete.
 * The destructive delete is routed to the sidebar's shared confirm dialog via
 * ``onDelete`` so there is one delete UX across kinds.
 */
export function TaggerSessionRow({
  item,
  isActive,
  onDelete,
  onRefresh,
}: {
  item: SidebarJobItem;
  isActive: boolean;
  onDelete: (e: React.MouseEvent, id: string) => void;
  onRefresh: () => void;
}) {
  const [menuOpen, setMenuOpen] = React.useState(false);
  const [renaming, setRenaming] = React.useState(false);
  const [renameValue, setRenameValue] = React.useState("");
  const [menuPos, setMenuPos] = React.useState<{ top: number; left: number } | null>(null);
  const menuRef = React.useRef<HTMLDivElement>(null);
  const btnRef = React.useRef<HTMLButtonElement>(null);
  const renameRef = React.useRef<HTMLInputElement>(null);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  const id = item.optimization_id;
  const displayName = item.name?.trim() || msg("tagger.session.untitled");
  const rowCount = item.row_count ?? 0;
  const taggedCount = item.tagged_count ?? 0;

  React.useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: MouseEvent | TouchEvent) => {
      const target = e.target as Node;
      if (menuRef.current?.contains(target)) return;
      if (dropdownRef.current?.contains(target)) return;
      setMenuOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMenuOpen(false);
        btnRef.current?.focus();
      }
    };
    document.addEventListener("mousedown", handler);
    document.addEventListener("touchstart", handler);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", handler);
      document.removeEventListener("touchstart", handler);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [menuOpen]);

  React.useEffect(() => {
    if (renaming) renameRef.current?.focus();
  }, [renaming]);

  // Enter blurs the input, which fires onBlur → handleRename again; guard the
  // double-fire (same pattern as JobRow).
  const renameSubmittedRef = React.useRef(false);
  const handleRename = async () => {
    if (renameSubmittedRef.current) return;
    renameSubmittedRef.current = true;
    const newName = renameValue.trim();
    if (!newName || newName === (item.name ?? "")) {
      setRenaming(false);
      renameSubmittedRef.current = false;
      return;
    }
    try {
      await renameTaggerSession(id, newName);
      toast.success(msg("sidebar.rename.success"));
      window.dispatchEvent(new Event(TAGGER_SESSIONS_CHANGED));
      onRefresh();
    } catch {
      toast.error(msg("sidebar.rename.failed"));
    }
    setRenaming(false);
    renameSubmittedRef.current = false;
  };

  const handlePin = async () => {
    try {
      const res = await setTaggerSessionPinned(id, !item.pinned);
      toast.success(res.pinned ? msg("sidebar.pin.on") : msg("sidebar.pin.off"));
      window.dispatchEvent(new Event(TAGGER_SESSIONS_CHANGED));
      onRefresh();
    } catch {
      toast.error(msg("sidebar.generic_error"));
    }
    setMenuOpen(false);
  };

  if (renaming) {
    return (
      <div className="px-2 py-1.5">
        <input
          ref={renameRef}
          type="text"
          value={renameValue}
          onChange={(e) => setRenameValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void handleRename();
            }
            if (e.key === "Escape") {
              renameSubmittedRef.current = true;
              setRenaming(false);
              renameSubmittedRef.current = false;
            }
          }}
          onBlur={handleRename}
          maxLength={120}
          className="w-full text-[0.6875rem] bg-sidebar-accent/30 border border-primary/30 rounded-md px-2 py-1 outline-none font-medium"
          dir="auto"
        />
      </div>
    );
  }

  return (
    <div className="relative" ref={menuRef}>
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-lg px-2 py-2 text-[0.6875rem] transition-all duration-150",
          isActive
            ? "bg-primary/[0.07] text-foreground"
            : "text-muted-foreground hover:bg-sidebar-accent/30 hover:text-foreground",
        )}
      >
        <Link href={`/tagger/${id}`} className="flex items-center gap-2 min-w-0 flex-1 overflow-hidden">
          <span
            className="truncate font-medium leading-tight min-w-0 block text-start flex-1"
            title={displayName}
          >
            {displayName}
          </span>
          {rowCount > 0 && (
            <span className="shrink-0 text-[9px] font-semibold text-muted-foreground/60 bg-muted/40 px-1 py-0.5 rounded tabular-nums">
              {taggedCount}/{rowCount}
            </span>
          )}
          {item.pinned && <Pin className="size-2.5 text-muted-foreground/60 shrink-0" />}
        </Link>
        <button
          ref={btnRef}
          type="button"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!menuOpen && btnRef.current) {
              const rect = btnRef.current.getBoundingClientRect();
              const menuWidth = 140;
              const menuHeightEstimate = 140;
              const margin = 8;
              const left = Math.max(
                margin,
                Math.min(rect.right - menuWidth, window.innerWidth - menuWidth - margin),
              );
              const top =
                rect.bottom + menuHeightEstimate + margin > window.innerHeight
                  ? Math.max(margin, rect.top - menuHeightEstimate - 4)
                  : rect.bottom + 4;
              setMenuPos({ top, left });
            }
            setMenuOpen((o) => !o);
          }}
          className="p-0.5 rounded cursor-pointer text-muted-foreground/40 hover:text-foreground transition-colors shrink-0"
          aria-label={formatMsg("auto.features.sidebar.components.sidebar.template.3", {
            p1: displayName,
          })}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
        >
          <MoreHorizontal className="size-3.5" />
        </button>
      </div>

      {menuOpen &&
        menuPos &&
        createPortal(
          <motion.div
            ref={dropdownRef}
            role="menu"
            initial={{ opacity: 0, scale: 0.95, y: -4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -4 }}
            transition={{ duration: 0.12 }}
            className="fixed z-[9999] min-w-[140px] rounded-2xl border border-border/40 bg-card shadow-[0_4px_24px_rgba(28,22,18,0.1)] py-1.5"
            style={{ top: menuPos.top, left: menuPos.left, right: "auto" }}
          >
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setMenuOpen(false);
                setRenameValue(item.name ?? displayName);
                setRenaming(true);
              }}
              className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors"
            >
              <Pencil className="size-3.5 text-muted-foreground" />
              {msg("auto.features.sidebar.components.sidebar.8")}
            </button>

            <button
              type="button"
              role="menuitem"
              onClick={handlePin}
              className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors"
            >
              <Pin className={cn("size-3.5", item.pinned ? "text-foreground" : "text-muted-foreground")} />
              {item.pinned
                ? msg("auto.features.sidebar.components.sidebar.literal.13")
                : msg("auto.features.sidebar.components.sidebar.literal.14")}
            </button>

            <div className="h-px bg-border/20 mx-2 my-1" />

            <button
              type="button"
              role="menuitem"
              onClick={(e) => {
                setMenuOpen(false);
                onDelete(e, id);
              }}
              className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-red-500 hover:bg-red-500/5 cursor-pointer transition-colors"
            >
              <Trash2 className="size-3.5" />
              {msg("auto.features.sidebar.components.sidebar.10")}
            </button>
          </motion.div>,
          document.body,
        )}
    </div>
  );
}
