import type { Metadata } from "next";

import { msg } from "@/shared/lib/messages";

// Public share pages must never be indexed by search engines — they expose a
// (scrubbed) read-only optimization to anyone with the link, not to crawlers.
export function generateMetadata(): Metadata {
  return {
    title: msg("share.page.title"),
    robots: { index: false, follow: false },
  };
}

export default function ShareLayout({ children }: { children: React.ReactNode }) {
  return children;
}
