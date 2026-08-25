"use client";

import { Suspense } from "react";
import { DatasetEditorView } from "@/features/datasets";

export default function DatasetEditPage() {
  return (
    <Suspense fallback={null}>
      <DatasetEditorView />
    </Suspense>
  );
}
