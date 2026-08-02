"use client";

import { useState } from "react";
import { CircleNotch, Trash } from "@/shared/ui/icons";
import { toast } from "react-toastify";
import { Button } from "@/shared/ui/primitives/button";
import { Dialog, DialogContent, DialogFooter } from "@/shared/ui/primitives/dialog";
import { DialogTitleRow } from "@/shared/ui/dialog-title-row";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { deleteJob } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";

export function DeleteJobDialog({
  optimizationId,
  onDeleted,
}: {
  optimizationId: string;
  onDeleted: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const handleDelete = async () => {
    setLoading(true);
    try {
      await deleteJob(optimizationId);
      setOpen(false);
      window.dispatchEvent(new Event("optimizations-changed"));
      onDeleted();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("optimization.delete.failed"));
    } finally {
      setLoading(false);
    }
  };
  return (
    <>
      <TooltipButton tooltip={msg("auto.features.optimizations.components.deletejobdialog.1")}>
        <Button
          variant="ghost"
          size="icon"
          className="size-8 text-muted-foreground hover:text-red-600"
          onClick={() => setOpen(true)}
          aria-label={msg("auto.features.optimizations.components.deletejobdialog.literal.1")}
        >
          <Trash className="size-4" />
        </Button>
      </TooltipButton>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md">
          <DialogTitleRow
            title={
              <>
                {msg("auto.features.optimizations.components.deletejobdialog.2")}
                {TERMS.optimization}
              </>
            }
            description={
              <>
                {msg("auto.features.optimizations.components.deletejobdialog.3")}
                {TERMS.optimization}{" "}
                <span className="font-mono font-semibold text-foreground break-all">
                  {optimizationId}
                </span>
                ?{" "}
                {msg("delete.irreversible")}
              </>
            }
          />

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setOpen(false)}
              disabled={loading}
              className="w-full justify-center"
            >
              {msg("auto.features.optimizations.components.deletejobdialog.4")}
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={loading}
              className="w-full justify-center"
            >
              {loading ? (
                <CircleNotch className="size-4 animate-spin" />
              ) : (
                msg("auto.features.optimizations.components.deletejobdialog.literal.2")
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
