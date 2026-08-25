"use client";

import * as React from "react";

import { cn } from "@/shared/lib/utils";

interface AgentThreadProps {
  children: React.ReactNode;
  scrollDeps?: readonly unknown[];
  emptyState?: React.ReactNode;
  isEmpty?: boolean;
  className?: string;
}

// Within this many px of the bottom still counts as "at the bottom", so
// momentum scrolling and fractional positions don't break follow mode.
const STICKY_BOTTOM_SLACK_PX = 48;

export function AgentThread({
  children,
  scrollDeps = [],
  emptyState,
  isEmpty,
  className,
}: AgentThreadProps) {
  const scrollRef = React.useRef<HTMLDivElement>(null);
  // Follow the stream only while the user is at the bottom. Once they scroll
  // up to read history their position persists — new tokens and re-renders
  // must not yank them back down. Scrolling back to the bottom re-engages
  // following. The programmatic pin below lands at distance 0, so it keeps
  // the flag true rather than fighting the user.
  const stickToBottomRef = React.useRef(true);

  const handleScroll = React.useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < STICKY_BOTTOM_SLACK_PX;
  }, []);

  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el || !stickToBottomRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [children, scrollDeps]);

  // Siblings mounting below the thread (answer choices, a growing composer)
  // shrink this box AFTER the effect above ran, sliding the last lines — the
  // question the user must answer — out of view. Follow mode re-pins on any
  // box resize so the bottom stays the bottom.
  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      if (stickToBottomRef.current) el.scrollTop = el.scrollHeight;
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    // No overscroll containment: once the thread hits its boundary (or is too
    // short to scroll at all), the wheel falls through and scrolls the page —
    // the thread must not be a dead zone for page scrolling.
    <div
      ref={scrollRef}
      onScroll={handleScroll}
      className={cn("flex-1 min-h-0 overflow-y-auto px-4 py-4 space-y-5", className)}
    >
      {isEmpty && emptyState}
      {children}
    </div>
  );
}
