"use client";

import * as React from "react";
import Link from "next/link";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Button } from "@/shared/ui/primitives/button";
import { msg } from "@/shared/lib/messages";
import { I18N_KEY, tI18n } from "@/shared/lib/i18n";
import { INSUFFICIENT_CREDITS_EVENT } from "@/shared/lib/api";

/**
 * Global paywall for the credit gate. Mounted once at the app root: the central
 * `request()` path dispatches {@link INSUFFICIENT_CREDITS_EVENT} when a managed
 * submit is refused with a 402, and this opens the single "add credits" modal —
 * so every blocked flow surfaces the same paywall instead of a per-call toast
 * (submit producers suppress their own toast via `isInsufficientCreditsError`).
 *
 * The body copy is the backend's own gate message — it already names the way
 * out (add credits) — so there is nothing to fetch; the modal is purely
 * presentational.
 */
export function InsufficientCreditsModalHost() {
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    const onBlocked = () => setOpen(true);
    window.addEventListener(INSUFFICIENT_CREDITS_EVENT, onBlocked);
    return () => window.removeEventListener(INSUFFICIENT_CREDITS_EVENT, onBlocked);
  }, []);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{msg("billing.upgrade.title")}</DialogTitle>
          <DialogDescription>{tI18n(I18N_KEY.BILLING_INSUFFICIENT_CREDITS)}</DialogDescription>
        </DialogHeader>

        <Button asChild onClick={() => setOpen(false)} className="w-full">
          <Link href="/upgrade">{msg("billing.action.add_credits")}</Link>
        </Button>
      </DialogContent>
    </Dialog>
  );
}
