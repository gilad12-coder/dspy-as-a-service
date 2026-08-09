import type { Metadata } from "next";

import { msg } from "@/shared/lib/messages";

// generateMetadata, not a static `metadata` object: the description resolves
// i18n, which must follow the request locale rather than freeze at module load.
export function generateMetadata(): Metadata {
  return {
    title: msg("storage.page.title"),
    description: msg("storage.page.subtitle"),
  };
}

export default function StorageLayout({ children }: { children: React.ReactNode }) {
  return children;
}
