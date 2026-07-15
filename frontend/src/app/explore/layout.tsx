import type { Metadata } from "next";
import { TERMS } from "@/shared/lib/terms";

import { formatMsg } from "@/shared/lib/messages";
// generateMetadata, not a static `metadata` object: the description resolves
// i18n, which must follow the request locale rather than freeze at module load.
export function generateMetadata(): Metadata {
  return {
    title: "Explore",
    description: formatMsg("auto.app.explore.layout.template.1", { p1: TERMS.optimizationPlural }),
  };
}

export default function ExploreLayout({ children }: { children: React.ReactNode }) {
  return children;
}
