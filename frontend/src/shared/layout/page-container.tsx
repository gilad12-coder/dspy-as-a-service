import type { ReactNode } from "react";

import { cn } from "@/shared/lib/utils";

/**
 * The app shell's standard page box: capped width, centered, with the shared
 * fluid inline padding. The shell wraps every route in it except /tagger,
 * whose surfaces pick their own width per phase — the session chooser renders
 * the capped box (identical geometry to /datasets, so the Data hub tabs never
 * shift between tabs) while the annotation surfaces render it `full`.
 *
 * The `.mx-auto.max-w-7xl` media rules in globals.css key off the capped
 * variant's classes; padding stays inline-style so those rules can widen the
 * cap without double-applying padding.
 */
export function PageContainer({ full = false, children }: { full?: boolean; children: ReactNode }) {
  return (
    <div
      className={cn("relative z-[1] py-6 md:py-8", full ? "max-w-none" : "mx-auto max-w-7xl")}
      style={{ paddingInline: "clamp(1rem, 5vw - 0.5rem, 2rem)" }}
    >
      {children}
    </div>
  );
}
