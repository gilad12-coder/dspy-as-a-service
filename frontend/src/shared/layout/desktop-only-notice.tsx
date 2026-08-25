"use client";

import { Desktop } from "@/shared/ui/icons";
import { EmptyState } from "@/shared/ui/empty-state";
import { msg } from "@/shared/lib/messages";

/**
 * Full-page stand-in for authoring routes in the phone shell (new
 * optimization, datasets, tagging, storage). Deliberately has no "continue
 * anyway" escape: those surfaces are desk work, and the phone shell exists to
 * keep the viewing paths fast rather than to squeeze every desktop feature in.
 */
export function DesktopOnlyNotice() {
  return (
    <div className="flex min-h-[60dvh] items-center justify-center">
      <EmptyState
        icon={Desktop}
        iconWrap="tile"
        variant="compact"
        title={msg("mobile.desktop_only.title")}
        description={msg("mobile.desktop_only.body")}
        action={{ label: msg("mobile.desktop_only.home_cta"), href: "/" }}
        className="rounded-2xl border border-border/40 bg-card/60"
      />
    </div>
  );
}
