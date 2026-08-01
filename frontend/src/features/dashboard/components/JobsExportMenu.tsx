import { Download } from "lucide-react";
import { toast } from "react-toastify";
import { Popover as PopoverPrimitive } from "radix-ui";
import { Button } from "@/shared/ui/primitives/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { msg } from "@/shared/lib/messages";
import type { OptimizationSummaryResponse } from "@/shared/types/api";
import { exportJobs, type JobsExportFormat } from "../lib/export-jobs";

const FORMATS: readonly JobsExportFormat[] = ["csv", "json", "xlsx", "xls"];

export function JobsExportMenu({
  jobs,
  showSharedColumns,
}: {
  jobs: OptimizationSummaryResponse[];
  showSharedColumns: boolean;
}) {
  const handleExport = async (format: JobsExportFormat) => {
    try {
      await exportJobs(jobs, showSharedColumns, format);
    } catch {
      toast.error(msg("optimization.file.parse_error"));
    }
  };

  return (
    <PopoverPrimitive.Root>
      <Tooltip>
        <TooltipTrigger asChild>
          <PopoverPrimitive.Trigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              disabled={jobs.length === 0}
              aria-label={msg("usage.action.export")}
            >
              <Download className="size-4" />
            </Button>
          </PopoverPrimitive.Trigger>
        </TooltipTrigger>
        <TooltipContent>{msg("usage.action.export")}</TooltipContent>
      </Tooltip>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          side="bottom"
          align="start"
          sideOffset={8}
          className="z-50 w-44 rounded-lg border bg-background p-1 shadow-lg animate-in fade-in-0 zoom-in-95"
        >
          {FORMATS.map((format) => (
            <PopoverPrimitive.Close key={format} asChild>
              <button
                type="button"
                onClick={() => void handleExport(format)}
                className="flex w-full items-center rounded-md px-3 py-1.5 text-xs font-medium text-foreground cursor-pointer transition-colors hover:bg-accent"
              >
                {format.toUpperCase()}
              </button>
            </PopoverPrimitive.Close>
          ))}
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}
