"use client";

import Link from "next/link";
import { ArrowLeft } from "@/shared/ui/icons";

import { msg } from "@/shared/lib/messages";
import { clearRecentSession } from "@/shared/lib/recent-session";

const LINK_CLASSES =
  "group inline-flex cursor-pointer items-center gap-1.5 rounded-md py-1 text-[13px] font-medium text-muted-foreground transition-colors duration-150 hover:text-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50";

const ARROW_CLASSES =
  "size-3.5 rtl:rotate-180 transition-transform duration-150 ease-out group-hover:-translate-x-0.5 rtl:group-hover:translate-x-0.5 motion-reduce:transition-none motion-reduce:transform-none";

/**
 * The quiet destination-labeled back link to the labeling-sessions list, shown
 * above every tagger session screen. By default it is a real ``<Link>`` to
 * ``/tagger`` that drops the resume mark — a deliberate exit, so the sidebar's
 * Text-tagging button offers a fresh start instead of bouncing back in. Pass
 * ``onExit`` for sessions still living on ``/tagger`` itself (the pre-persist
 * wizard flow), where "back" must reset local state rather than navigate. Pass
 * ``label`` to relabel the destination — e.g. a finished session's row view
 * points back to its own results overview, not out to the sessions list.
 */
export function TaggerBackLink({ onExit, label }: { onExit?: () => void; label?: string }) {
  const text = label ?? msg("tagger.session.back");
  if (onExit) {
    return (
      <button type="button" onClick={onExit} className={LINK_CLASSES}>
        <ArrowLeft className={ARROW_CLASSES} aria-hidden="true" />
        {text}
      </button>
    );
  }
  return (
    <Link href="/tagger" onClick={() => clearRecentSession("tagger")} className={LINK_CLASSES}>
      <ArrowLeft className={ARROW_CLASSES} aria-hidden="true" />
      {text}
    </Link>
  );
}
