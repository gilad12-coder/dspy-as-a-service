"use client";

import { useState } from "react";
import { Image as ImageIcon, Books, TextT as TypeIcon, UploadSimple } from "@/shared/ui/icons";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/shared/ui/primitives/card";
import { Button } from "@/shared/ui/primitives/button";
import { Label } from "@/shared/ui/primitives/label";
import { Badge } from "@/shared/ui/primitives/badge";
import { Separator } from "@/shared/ui/primitives/separator";
import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";
import { msg } from "@/shared/lib/messages";
import { DatasetPickerDialog } from "@/features/datasets";

import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";

export function DatasetStep({ w }: { w: SubmitWizardContext }) {
  const {
    parsedDataset,
    datasetFileName,
    fileInputRef,
    handleFileUpload,
    handlePickFromLibrary,
    columnRoles,
    setColumnRoles,
    columnKinds,
    setColumnKinds,
    datasetProfile,
  } = w;
  const [pickerOpen, setPickerOpen] = useState(false);

  // Auto-detected kinds straight from the profiler — used to mark a column
  // as "auto-detected as image" (vs a user-driven manual flip) in the UI.
  const autoDetectedKinds = new Map(
    (datasetProfile?.inputs ?? []).map((entry) => [entry.name, entry.kind]),
  );

  // Columns render in the dataset's own order — exactly as they appear in the
  // uploaded file (and, for a clone, the order they were submitted in). No
  // role-based reordering, so the on-screen order always mirrors the data.

  return (
    <Card
      className=" border-border/50 bg-card/80 backdrop-blur-xl shadow-lg"
      data-tutorial="wizard-step-2"
    >
      <CardHeader className="px-4 sm:px-6">
        <CardTitle className="text-lg">{TERMS.dataset}</CardTitle>
        <CardDescription>
          {msg("auto.features.submit.components.steps.datasetstep.1")}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5 px-4 sm:px-6">
        <label
          data-tutorial="dataset-upload"
          className={cn(
            "group block cursor-pointer rounded-xl border-2 border-dashed p-6 text-center transition-all duration-300 sm:p-10",
            parsedDataset
              ? "border-primary/40 bg-primary/5"
              : "hover:border-primary/50 hover:bg-muted/30",
          )}
        >
          <UploadSimple className="h-10 w-10 mx-auto mb-3 text-muted-foreground group-hover:text-primary/70 transition-colors duration-300" />
          <p
            className="text-sm font-medium truncate max-w-full px-4"
            title={datasetFileName ?? undefined}
          >
            {datasetFileName ?? msg("auto.features.submit.components.steps.datasetstep.literal.1")}
          </p>
          {parsedDataset && (
            <Badge variant="secondary" className="mt-2">
              {parsedDataset.rowCount}
              {msg("auto.features.submit.components.steps.datasetstep.2")}
              {parsedDataset.columns.length}
              {msg("auto.features.submit.components.steps.datasetstep.3")}
            </Badge>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.json,.xlsx,.xls"
            className="hidden"
            onChange={handleFileUpload}
          />
        </label>

        <div className="flex items-center gap-3">
          <Separator className="flex-1" />
          <span className="text-xs text-muted-foreground">{msg("submit.dataset.library_or")}</span>
          <Separator className="flex-1" />
        </div>

        <Button
          type="button"
          variant="outline"
          onClick={() => setPickerOpen(true)}
          className="min-h-[44px] w-full justify-center gap-2 lg:min-h-0"
        >
          <Books className="size-4" />
          {msg("submit.dataset.library_pick")}
        </Button>

        <DatasetPickerDialog
          open={pickerOpen}
          onOpenChange={setPickerOpen}
          onPick={handlePickFromLibrary}
        />

        {parsedDataset && parsedDataset.columns.length > 0 && (
          <>
            <Separator />
            <div className="space-y-3" data-tutorial="column-mapping">
              <Label>{msg("auto.features.submit.components.steps.datasetstep.4")}</Label>
              <p className="text-xs text-muted-foreground">
                {msg("auto.features.submit.components.steps.datasetstep.5")}
              </p>
              <div className="space-y-2">
                {parsedDataset.columns.map((col) => {
                  const isInput = columnRoles[col] === "input";
                  const kind = columnKinds[col] ?? "text";
                  const wasAutoImage = autoDetectedKinds.get(col) === "image";
                  return (
                    <div
                      key={col}
                      className="flex flex-col items-stretch gap-2 sm:flex-row sm:items-center sm:justify-between"
                    >
                      <div className="flex min-w-0 flex-1 items-center gap-2">
                        <span className="text-xs sm:text-sm font-mono truncate" dir="ltr">
                          {col}
                        </span>
                        {isInput && (
                          <button
                            type="button"
                            onClick={() =>
                              setColumnKinds((prev) => ({
                                ...prev,
                                [col]: kind === "image" ? "text" : "image",
                              }))
                            }
                            className={cn(
                              "inline-flex min-h-[44px] shrink-0 cursor-pointer items-center gap-1 rounded-md border px-2 py-1 text-[0.625rem] font-medium transition-colors sm:px-1.5 sm:py-0.5 lg:min-h-0",
                              kind === "image"
                                ? "border-primary/40 bg-primary/10 text-primary hover:bg-primary/15"
                                : "border-border/60 bg-muted/40 text-muted-foreground hover:border-primary/30 hover:text-foreground",
                            )}
                            title={
                              kind === "image"
                                ? wasAutoImage
                                  ? msg("submit.dataset.column_kind.image_auto_hint")
                                  : msg("submit.dataset.column_kind.image")
                                : msg("submit.dataset.column_kind.text_manual_hint")
                            }
                          >
                            {kind === "image" ? (
                              <ImageIcon className="size-3" />
                            ) : (
                              <TypeIcon className="size-3" />
                            )}
                            <span>
                              {kind === "image"
                                ? msg("submit.dataset.column_kind.image")
                                : msg("submit.dataset.column_kind.text")}
                            </span>
                          </button>
                        )}
                      </div>
                      {(() => {
                        const options = [
                          [
                            "input",
                            msg("auto.features.submit.components.steps.datasetstep.literal.2"),
                          ],
                          [
                            "output",
                            msg("auto.features.submit.components.steps.datasetstep.literal.3"),
                          ],
                          [
                            "ignore",
                            msg("auto.features.submit.components.steps.datasetstep.literal.4"),
                          ],
                        ] as const;
                        const activeIdx = options.findIndex(([v]) => v === columnRoles[col]);
                        // Track = 2px padding + 2px gaps, so each flex-1 button is
                        // (100% - 8px)/3 wide and steps by (100% - 2px)/3. The pill must
                        // match that stride or it drifts further off per segment.
                        const pillLeft =
                          activeIdx >= 0 ? `calc(2px + ${activeIdx} * (100% - 2px) / 3)` : "2px";
                        return (
                          <div
                            // Arbitrary column syntax on purpose: the literal
                            // `grid-cols-3` class is force-stacked to one column
                            // by the global mobile rule in globals.css, which
                            // would break the sliding pill's horizontal math.
                            className="relative inline-grid w-full shrink-0 [grid-template-columns:repeat(3,minmax(0,1fr))] gap-0.5 rounded-lg bg-muted p-0.5 sm:w-auto"
                          >
                            <div
                              className="absolute top-0.5 bottom-0.5 rounded-md bg-stone-500/15 shadow-sm transition-[inset-inline-start] duration-100 ease-out"
                              style={{
                                width: "calc((100% - 8px) / 3)",
                                insetInlineStart: pillLeft,
                              }}
                            />
                            {options.map(([val, label]) => (
                              <button
                                key={val}
                                type="button"
                                onClick={() => setColumnRoles((prev) => ({ ...prev, [col]: val }))}
                                className={cn(
                                  "relative z-10 min-h-[44px] cursor-pointer rounded-md px-2 py-1 text-center text-xs font-medium transition-colors duration-100 sm:px-3 lg:min-h-0",
                                  columnRoles[col] === val
                                    ? "text-stone-600"
                                    : "text-muted-foreground hover:text-foreground",
                                )}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        );
                      })()}
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
