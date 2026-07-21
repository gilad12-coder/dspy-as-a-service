"use client";

import * as React from "react";
import { useGeneralistPanelStateOptional } from "../hooks/use-panel-state";

/**
 * Re-homes the floating agent pill for as long as this is mounted: the pill
 * portals in here instead of overlaying the viewport corner. Full-bleed
 * working surfaces whose bottom action bar owns that corner (the tagger
 * walkthroughs' Prev/Next row) render one where the launcher should sit —
 * the app-chrome equivalent of swapping a chat widget's default launcher for
 * a custom docked one. Renders nothing when the agent feature is disabled.
 */
export function AgentPillDock({ className }: { className?: string }) {
  const panel = useGeneralistPanelStateOptional();
  const [el, setEl] = React.useState<HTMLSpanElement | null>(null);
  const register = panel?.registerPillDock;
  React.useEffect(() => {
    if (!register || !el) return;
    return register(el);
  }, [register, el]);
  if (!panel) return null;
  return <span ref={setEl} className={className} />;
}
