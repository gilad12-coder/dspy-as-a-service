"use client";

import * as React from "react";
import { useSearchParams } from "next/navigation";
import { Database, Search, Upload } from "lucide-react";
import { toast } from "react-toastify";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { EmptyState } from "@/shared/ui/empty-state";
import { isStorageQuotaError, saveDataset, type DatasetSummary } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { parseDatasetFile } from "@/shared/lib/parse-dataset";
import { cn } from "@/shared/lib/utils";
import { useDatasets } from "../hooks/use-datasets";
import { DatasetCard } from "./DatasetCard";
import { DatasetDetailDialog } from "./DatasetDetailDialog";
import { DatasetsSkeleton } from "./DatasetsSkeleton";

const UPLOAD_ACCEPT = ".csv,.json,.xlsx,.xls";

/**
 * Top-level /datasets page: the personal dataset library. Lists owned and
 * shared-in datasets as searchable cards over a usage meter, with a drag-in /
 * click upload that parses CSV/JSON/XLSX and saves a new entry. Selecting a card
 * (or arriving with ``?open=<id>``) opens a read-only detail sheet with a row
 * preview and the reverse link to every optimization that used the dataset.
 */
export function DatasetsView() {
  const { datasets, loading, error, refetch } = useDatasets();
  const searchParams = useSearchParams();
  const [search, setSearch] = React.useState("");
  const [selected, setSelected] = React.useState<DatasetSummary | null>(null);
  const [dragging, setDragging] = React.useState(false);
  const [uploading, setUploading] = React.useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const deepLinkedRef = React.useRef(false);

  // Honour ?open=<id> once the list has loaded: open that dataset's detail sheet
  // (the navigable link from an optimization's source-dataset row). Guarded so
  // it fires a single time, not again after the user closes the sheet. When the
  // id resolves to nothing — the source dataset was deleted or unshared — say so
  // rather than dead-ending silently on the click.
  React.useEffect(() => {
    if (deepLinkedRef.current || loading || error) return;
    const openId = searchParams.get("open");
    if (!openId) return;
    deepLinkedRef.current = true;
    const match = datasets.find((d) => d.id === openId);
    if (match) setSelected(match);
    else toast.info(msg("datasets.open.not_found"));
  }, [datasets, loading, error, searchParams]);

  const filtered = React.useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return datasets;
    return datasets.filter((d) => d.name.toLowerCase().includes(q));
  }, [datasets, search]);

  const handleFiles = React.useCallback(
    async (files: FileList | null) => {
      const file = files?.[0];
      if (!file || uploading) return;
      setUploading(true);
      try {
        const parsed = await parseDatasetFile(file);
        const name = file.name.replace(/\.[^.]+$/, "") || file.name;
        await saveDataset({
          name,
          source: "upload",
          dataset: parsed.rows,
          column_schema: { column_order: parsed.columns },
        });
        toast.success(msg("datasets.toast.uploaded"));
        refetch();
      } catch (err) {
        // The storage-budget 409 opens the shared quota modal centrally, so
        // suppress the redundant toast here (and at the other save producers).
        if (!isStorageQuotaError(err)) {
          toast.error(err instanceof Error ? err.message : msg("datasets.toast.upload_failed"));
        }
      } finally {
        setUploading(false);
      }
    },
    [uploading, refetch],
  );

  if (loading) return <DatasetsSkeleton />;

  return (
    <div className="pb-16">
      <input
        ref={fileInputRef}
        type="file"
        accept={UPLOAD_ACCEPT}
        className="hidden"
        onChange={(e) => {
          void handleFiles(e.target.files);
          e.target.value = "";
        }}
      />

      <div className="flex items-center gap-2.5">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute end-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={msg("datasets.search.placeholder")}
            aria-label={msg("datasets.search.placeholder")}
            className="pe-9"
          />
        </div>
        <Button
          variant="outline"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          className="shrink-0"
        >
          <Upload className="size-4" />
          {msg("datasets.upload")}
        </Button>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!dragging) setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          if (e.currentTarget === e.target) setDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void handleFiles(e.dataTransfer.files);
        }}
        className={cn(
          "mt-5 rounded-xl border border-dashed transition-colors duration-150",
          dragging
            ? "border-[#3D2E22]/50 bg-[#3D2E22]/[0.03]"
            : "border-transparent",
        )}
      >
        {error ? (
          <EmptyState icon={Database} title={msg("datasets.error")} />
        ) : datasets.length === 0 ? (
          <EmptyState
            icon={Database}
            iconWrap="tile"
            title={msg("datasets.empty.title")}
            description={msg("datasets.empty.body")}
            action={{ label: msg("datasets.upload"), onClick: () => fileInputRef.current?.click() }}
          />
        ) : filtered.length === 0 ? (
          <EmptyState icon={Search} title={msg("datasets.search.empty")} />
        ) : (
          <div className="flex flex-col gap-2.5 p-0.5">
            {dragging && (
              <p className="pointer-events-none py-2 text-center text-sm font-medium text-[#3D2E22]/70">
                {msg("datasets.upload.drop")}
              </p>
            )}
            {filtered.map((dataset) => (
              <DatasetCard
                key={dataset.id}
                dataset={dataset}
                onOpen={setSelected}
                onChanged={refetch}
              />
            ))}
          </div>
        )}
      </div>

      <DatasetDetailDialog dataset={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
