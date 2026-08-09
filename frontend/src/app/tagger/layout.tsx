import type { Metadata } from "next";
import { Suspense } from "react";
import { CircleNotch } from "@/shared/ui/icons";
import { TERMS } from "@/shared/lib/terms";

import { formatMsg, msg } from "@/shared/lib/messages";
import { PageContainer } from "@/shared/layout/page-container";
// generateMetadata, not a static `metadata` object: the description resolves
// i18n, which must follow the request locale rather than freeze at module load.
export function generateMetadata(): Metadata {
  return {
    title: msg("tagger.page.title"),
    description: formatMsg("auto.app.tagger.layout.template.1", { p1: TERMS.dataset }),
  };
}

export default function TaggerLayout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <PageContainer full>
          <div className="flex items-center justify-center min-h-[60vh]">
            <CircleNotch className="size-8 animate-spin text-primary" />
          </div>
        </PageContainer>
      }
    >
      {children}
    </Suspense>
  );
}
