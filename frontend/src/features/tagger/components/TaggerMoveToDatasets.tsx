"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "react-toastify";
import { ArrowLineRight, CircleNotch, Database } from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { Label } from "@/shared/ui/primitives/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { isStorageQuotaError, moveTaggerSessionToLibrary } from "@/shared/lib/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import { clearRecentSession } from "@/shared/lib/recent-session";
import { buildLibraryRows } from "../lib/export-csv";
import { TAGGER_SESSIONS_CHANGED } from "../hooks/use-tagger";
import type { Annotation, AnnotationProvenance, DataRow, TaggerConfig } from "../lib/types";

/**
 * The finished-session doorway into the Datasets tab. A banner CTA above the
 * results table that, on confirm, saves the labeled dataset to the library,
 * carries the session's sharing onto it, deletes the session, and lands the user
 * on /datasets — so no completed labeling session is left un-transitioned. The
 * move is owner-only and irreversible; the caller mounts this only when the
 * session is persisted and the caller owns it.
 */
export function TaggerMoveToDatasets({
  sessionId,
  config,
  data,
  columns,
  annotations,
  provenance,
}: {
  sessionId: string;
  config: TaggerConfig;
  data: DataRow[];
  columns: string[];
  annotations: Record<string, Annotation>;
  provenance?: Record<string, AnnotationProvenance>;
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [moving, setMoving] = useState(false);

  const defaultName = useCallback(
    () => `tagging_${config.mode}_${new Date().toISOString().slice(0, 10)}`,
    [config.mode],
  );

  const openDialog = useCallback(() => {
    setName(defaultName());
    setOpen(true);
  }, [defaultName]);

  const handleMove = useCallback(async () => {
    if (moving) return;
    const trimmed = name.trim() || defaultName();
    setMoving(true);
    try {
      const { rows, columnOrder, columnRoles } = buildLibraryRows(
        data,
        columns,
        annotations,
        config,
        provenance,
      );
      const res = await moveTaggerSessionToLibrary(sessionId, {
        name: trimmed,
        dataset: rows,
        column_schema: { column_order: columnOrder, column_roles: columnRoles },
      });
      // The session is gone: forget the resume mark so the sidebar starts fresh,
      // and refresh any mounted session list before navigating away.
      clearRecentSession("tagger");
      window.dispatchEvent(new Event(TAGGER_SESSIONS_CHANGED));
      toast.success(
        res.deduplicated
          ? msg("datasets.toast.deduplicated")
          : formatMsg("tagger.move.moved", { name: res.dataset.name }),
      );
      router.push("/datasets");
    } catch (err) {
      if (!isStorageQuotaError(err)) {
        toast.error(err instanceof Error ? err.message : msg("tagger.move.failed"));
      }
      setMoving(false);
    }
  }, [moving, name, defaultName, data, columns, annotations, config, provenance, sessionId, router]);

  return (
    <>
      <div className="flex flex-col gap-3 rounded-lg border border-border/60 bg-muted/30 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
            <Database className="size-5" />
          </span>
          <div className="min-w-0">
            <p className="text-sm font-medium text-foreground">{msg("tagger.move.title")}</p>
            <p className="text-sm text-muted-foreground">{msg("tagger.move.subtitle")}</p>
          </div>
        </div>
        <Button onClick={openDialog} className="shrink-0 gap-1.5 sm:self-center">
          <ArrowLineRight className="size-4 rtl:-scale-x-100" />
          {msg("tagger.move.cta")}
        </Button>
      </div>

      <Dialog open={open} onOpenChange={(v) => !moving && setOpen(v)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{msg("tagger.move.confirm_title")}</DialogTitle>
            <DialogDescription>{msg("tagger.move.confirm_body")}</DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="move-dataset-name">{msg("tagger.library.name_label")}</Label>
            <Input
              id="move-dataset-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={moving}
              dir="auto"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)} disabled={moving}>
              {msg("tagger.library.name_cancel")}
            </Button>
            <Button onClick={() => void handleMove()} disabled={moving} className="gap-1.5">
              {moving && <CircleNotch className="size-4 animate-spin" />}
              {msg("tagger.move.cta")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
