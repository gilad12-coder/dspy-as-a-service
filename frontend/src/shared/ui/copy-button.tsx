"use client";

import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, Copy } from "@/shared/ui/icons";

import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/primitives/button";

/**
 * Shared copy-to-clipboard state: `copy(text)` writes to the clipboard and
 * flips `copied` on for `resetMs` so the caller's glyph/label can show
 * success. Resolves to whether the write succeeded (clipboard access can be
 * denied); re-copying restarts the timer.
 */
export function useCopyToClipboard(resetMs = 1500) {
  const [copied, setCopied] = React.useState(false);
  const timer = React.useRef<number | null>(null);
  React.useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  const copy = React.useCallback(
    async (text: string) => {
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        return false;
      }
      setCopied(true);
      if (timer.current !== null) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), resetMs);
      return true;
    },
    [resetMs],
  );
  return { copied, copy };
}

/**
 * The app-standard copy glyph: a Copy icon that morphs into a check mark
 * with a quick scale/fade while `copied` is on. Use inside custom-styled
 * buttons; plain icon-only controls should use {@link CopyButton} instead.
 */
export function CopyGlyph({
  copied,
  className,
  checkClassName,
}: {
  copied: boolean;
  className?: string;
  checkClassName?: string;
}) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.span
        key={copied ? "check" : "copy"}
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.5, opacity: 0 }}
        transition={{ duration: 0.12, ease: "easeOut" }}
        className="inline-flex"
      >
        {copied ? (
          <Check className={cn(className, checkClassName)} aria-hidden="true" />
        ) : (
          <Copy className={className} aria-hidden="true" />
        )}
      </motion.span>
    </AnimatePresence>
  );
}

interface CopyButtonProps
  extends Omit<React.ComponentProps<typeof Button>, "onClick" | "children" | "aria-label"> {
  /** The clipboard payload. */
  text: string;
  ariaLabel: string;
  /** Optional label announced while the check mark is showing. */
  copiedAriaLabel?: string;
  /** Fired after a successful copy — e.g. to raise a toast. */
  onCopied?: () => void;
  /** Fired when clipboard access is denied. */
  onCopyError?: () => void;
  iconClassName?: string;
  /** Stop the click from reaching row/card handlers behind the button. */
  stopPropagation?: boolean;
}

/**
 * The app-standard copy button: an icon Button whose Copy glyph morphs into a
 * check mark after a successful copy. Extra props (including Radix tooltip
 * trigger injections) pass through to the underlying Button.
 */
export function CopyButton({
  text,
  ariaLabel,
  copiedAriaLabel,
  onCopied,
  onCopyError,
  variant = "ghost",
  size = "icon-sm",
  iconClassName = "size-3.5",
  stopPropagation,
  ...rest
}: CopyButtonProps) {
  const { copied, copy } = useCopyToClipboard();
  return (
    <Button
      variant={variant}
      size={size}
      {...rest}
      aria-label={copied && copiedAriaLabel ? copiedAriaLabel : ariaLabel}
      onClick={async (e) => {
        if (stopPropagation) e.stopPropagation();
        if (await copy(text)) onCopied?.();
        else onCopyError?.();
      }}
    >
      <CopyGlyph copied={copied} className={iconClassName} />
    </Button>
  );
}
