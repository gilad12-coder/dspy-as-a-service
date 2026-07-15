import type { Metadata } from "next";

import { msg } from "@/shared/lib/messages";

// generateMetadata, not a static `metadata` object: the description resolves
// i18n, which must follow the request locale rather than freeze at module load.
export function generateMetadata(): Metadata {
  return {
    title: "Datasets",
    description: msg("datasets.subtitle"),
  };
}

export default function DatasetsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
