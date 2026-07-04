import { Loader2 } from "lucide-react";
import { Button } from "@/shared/ui/primitives/button";
import { Dialog, DialogContent, DialogFooter } from "@/shared/ui/primitives/dialog";
import { DialogTitleRow } from "@/shared/ui/dialog-title-row";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";

type DeleteDialogsProps = {
  bulkDeleteOpen: boolean;
  setBulkDeleteOpen: (open: boolean) => void;
  bulkDeleting: boolean;
  confirmBulkDelete: () => void;
  selectedCount: number;
};

export function DeleteDialogs({
  bulkDeleteOpen,
  setBulkDeleteOpen,
  bulkDeleting,
  confirmBulkDelete,
  selectedCount,
}: DeleteDialogsProps) {
  return (
    <Dialog
      open={bulkDeleteOpen}
      onOpenChange={(open) => {
        if (!open) setBulkDeleteOpen(false);
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogTitleRow
          title={
            selectedCount === 1
              ? formatMsg("auto.features.dashboard.components.deletedialogs.template.1", {
                  p1: TERMS.optimization,
                })
              : formatMsg("auto.features.dashboard.components.deletedialogs.template.2", {
                  p1: TERMS.optimizationPlural,
                })
          }
          description={
            selectedCount === 1 ? (
              <>
                {msg("auto.features.dashboard.components.deletedialogs.1")}
                {TERMS.optimization}
                {msg("auto.features.dashboard.components.deletedialogs.2")}
              </>
            ) : (
              <>
                {msg("auto.features.dashboard.components.deletedialogs.3")}
                <span className="font-semibold text-foreground">{selectedCount}</span>{" "}
                {TERMS.optimizationPlural}
                {msg("auto.features.dashboard.components.deletedialogs.4")}
              </>
            )
          }
        />
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setBulkDeleteOpen(false)}
            disabled={bulkDeleting}
            className="w-full justify-center"
          >
            {msg("auto.features.dashboard.components.deletedialogs.5")}
          </Button>
          <Button
            variant="destructive"
            onClick={confirmBulkDelete}
            disabled={bulkDeleting}
            className="w-full justify-center"
          >
            {bulkDeleting ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              msg("auto.features.dashboard.components.deletedialogs.literal.1")
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
