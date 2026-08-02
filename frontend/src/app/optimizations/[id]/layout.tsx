import type { Metadata } from "next";
import { Suspense } from "react";
import { CircleNotch } from "@/shared/ui/icons";
import { TERMS } from "@/shared/lib/terms";

import { formatMsg } from "@/shared/lib/messages";
// generateMetadata, not a static `metadata` object: the description resolves
// i18n, which must follow the request locale rather than freeze at module load.
export function generateMetadata(): Metadata {
  return {
    title: "Optimization Details",
    description: formatMsg("auto.app.optimizations.id.layout.template.1", {
      p1: TERMS.optimization,
    }),
  };
}

export default function JobLayout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-[60vh]">
          <CircleNotch className="size-8 animate-spin text-primary" />
        </div>
      }
    >
      {children}
    </Suspense>
  );
}
