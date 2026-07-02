"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Loader2, Pencil, Pin, Tags, Trash2 } from "lucide-react";
import { toast } from "react-toastify";

import { Badge } from "@/shared/ui/primitives/badge";
import { Button } from "@/shared/ui/primitives/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Input } from "@/shared/ui/primitives/input";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import {
  deleteTaggerSession,
  renameTaggerSession,
  setTaggerSessionPinned,
  type TaggerSessionSummary,
} from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { formatRelativeTime } from "@/shared/lib/formatters";
import { cn } from "@/shared/lib/utils";
import { TAGGER_SESSIONS_CHANGED } from "../hooks/use-tagger";

/**
 * One saved text-labeling session on the Datasets page, styled to match
 * {@link DatasetCard}: an icon tile, the session name with a labeled/total
 * progress badge, and last-updated time. Clicking the card resumes annotating
 * at ``/tagger/[id]``; the trailing actions pin, rename, or delete it. Mutations
 * fire {@link TAGGER_SESSIONS_CHANGED} so any other open list refreshes too.
 */
export function TaggingSessionCard({
  session,
  onChanged,
}: {
  session: TaggerSessionSummary;
  onChanged: () => void;
}) {
  const router = useRouter();
  const [renameOpen, setRenameOpen] = React.useState(false);
  const [renameValue, setRenameValue] = React.useState(session.name);
  const [renaming, setRenaming] = React.useState(false);
  const [deleteOpen, setDeleteOpen] = React.useState(false);
  const [deleting, setDeleting] = React.useState(false);
  const [pinning, setPinning] = React.useState(false);

  const displayName = session.name?.trim() || msg("tagger.session.untitled");

  const notifyChanged = () => {
    window.dispatchEvent(new Event(TAGGER_SESSIONS_CHANGED));
    onChanged();
  };

  const resume = () => router.push(`/tagger/${session.id}`);

  const handleRename = async () => {
    const name = renameValue.trim();
    if (!name || renaming) return;
    setRenaming(true);
    try {
      await renameTaggerSession(session.id, name);
      setRenameOpen(false);
      notifyChanged();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("datasets.toast.rename_failed"));
    } finally {
      setRenaming(false);
    }
  };

  const handlePin = async () => {
    if (pinning) return;
    setPinning(true);
    try {
      await setTaggerSessionPinned(session.id, !session.pinned);
      notifyChanged();
    } catch {
      // Pin is a non-critical ordering hint; a failed toggle just leaves the
      // session where it was.
    } finally {
      setPinning(false);
    }
  };

  const handleDelete = async () => {
    if (deleting) return;
    setDeleting(true);
    try {
      await deleteTaggerSession(session.id);
      setDeleteOpen(false);
      notifyChanged();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("datasets.toast.delete_failed"));
    } finally {
      setDeleting(false);
    }
  };

  // Trailing-action clicks must not also resume the session.
  const stop = (e: React.MouseEvent) => e.stopPropagation();

  return (
    <>
      <div
        role="button"
        tabIndex={0}
        onClick={resume}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            resume();
          }
        }}
        aria-label={displayName}
        className="group flex cursor-pointer items-center gap-4 rounded-xl border border-[#DDD4C8]/60 bg-gradient-to-b from-white/95 to-[#F8F4EF] px-4 py-3.5 text-start shadow-[0_1px_3px_rgba(28,22,18,0.03)] transition-[border-color,box-shadow] duration-200 hover:border-[#C8B9A8]/70 hover:shadow-[0_2px_10px_rgba(28,22,18,0.06)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
      >
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[#3D2E22]/8 text-[#3D2E22]">
          <Tags className="size-5" />
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <p className="truncate text-sm font-semibold text-foreground">{displayName}</p>
            <Badge variant="secondary" size="sm" className="tabular-nums">
              {session.tagged_count}/{session.row_count}
            </Badge>
          </div>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">
            {formatRelativeTime(session.updated_at)}
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-1" onClick={stop}>
          <TooltipButton tooltip={session.pinned ? msg("tagger.session.unpin") : msg("tagger.session.pin")}>
            <Button
              variant="ghost"
              size="icon-sm"
              className={cn(
                "text-muted-foreground hover:text-foreground",
                session.pinned && "text-[#3D2E22]",
              )}
              onClick={handlePin}
              disabled={pinning}
              aria-label={session.pinned ? msg("tagger.session.unpin") : msg("tagger.session.pin")}
            >
              <Pin className={cn("size-4", session.pinned && "fill-current")} />
            </Button>
          </TooltipButton>
          <TooltipButton tooltip={msg("datasets.action.rename")}>
            <Button
              variant="ghost"
              size="icon-sm"
              className="text-muted-foreground hover:text-foreground"
              onClick={() => {
                setRenameValue(displayName);
                setRenameOpen(true);
              }}
              aria-label={msg("datasets.action.rename")}
            >
              <Pencil className="size-4" />
            </Button>
          </TooltipButton>
          <TooltipButton tooltip={msg("datasets.action.delete")}>
            <Button
              variant="ghost"
              size="icon-sm"
              className="text-muted-foreground hover:text-destructive"
              onClick={() => setDeleteOpen(true)}
              aria-label={msg("datasets.action.delete")}
            >
              <Trash2 className="size-4" />
            </Button>
          </TooltipButton>
        </div>
      </div>

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md">
          <DialogHeader className="text-start">
            <DialogTitle>{msg("tagger.session.rename_title")}</DialogTitle>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void handleRename();
              }
            }}
            aria-label={msg("datasets.rename.label")}
            autoFocus
          />
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setRenameOpen(false)}
              disabled={renaming}
              className="w-full justify-center"
            >
              {msg("datasets.rename.cancel")}
            </Button>
            <Button
              onClick={handleRename}
              disabled={renaming || renameValue.trim().length === 0}
              className="w-full justify-center shadow-xs"
            >
              {renaming ? <Loader2 className="size-4 animate-spin" /> : msg("datasets.rename.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md">
          <DialogHeader className="text-start">
            <DialogTitle>{msg("tagger.session.delete_title")}</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">{msg("tagger.session.delete_body")}</p>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteOpen(false)}
              disabled={deleting}
              className="w-full justify-center"
            >
              {msg("datasets.delete.cancel")}
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleting}
              className="w-full justify-center shadow-xs"
            >
              {deleting ? <Loader2 className="size-4 animate-spin" /> : msg("datasets.delete.confirm")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
