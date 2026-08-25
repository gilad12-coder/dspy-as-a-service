import type { Metadata } from "next";

import { msg } from "@/shared/lib/messages";

// Share pages must never be indexed — they expose a read-only dataset to anyone
// with the link, not to crawlers.
export function generateMetadata(): Metadata {
  return {
    title: msg("datasets.share.page.title"),
    robots: { index: false, follow: false },
  };
}

export default function DatasetShareLayout({ children }: { children: React.ReactNode }) {
  return children;
}
