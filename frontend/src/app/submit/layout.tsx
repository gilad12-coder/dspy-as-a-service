import type { Metadata } from "next";
import { Suspense } from "react";
import { CircleNotch } from "@/shared/ui/icons";
import { TERMS } from "@/shared/lib/terms";

import { formatMsg } from "@/shared/lib/messages";
// generateMetadata, not a static `metadata` object: the description resolves
// i18n, which must follow the request locale rather than freeze at module load.
export function generateMetadata(): Metadata {
  return {
    title: "New Optimization",
    description: formatMsg("auto.app.submit.layout.template.1", {
      p1: TERMS.model,
      p2: TERMS.dataset,
    }),
  };
}

export default function SubmitLayout({ children }: { children: React.ReactNode }) {
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
