"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, GraduationCap } from "lucide-react";
import { useTutorialContext } from "./tutorial-provider";
import { msg } from "@/shared/lib/messages";

export function TutorialMenu() {
  const { state, startTrack, closeMenu } = useTutorialContext();
  const startBtnRef = React.useRef<HTMLButtonElement | null>(null);
  const dialogRef = React.useRef<HTMLDivElement | null>(null);
  const titleId = React.useId();

  React.useEffect(() => {
    if (!state.isMenuOpen) return;
    startBtnRef.current?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        closeMenu();
        return;
      }
      // Trap focus inside the dialog. Without this, Tab walks out into
      // the page chrome behind the modal backdrop.
      if (e.key !== "Tab") return;
      const root = dialogRef.current;
      if (!root) return;
      const focusables = root.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (focusables.length === 0) return;
      const first = focusables[0]!;
      const last = focusables[focusables.length - 1]!;
      const active = document.activeElement;
      if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.isMenuOpen, closeMenu]);

  return (
    <AnimatePresence>
      {state.isMenuOpen && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0 bg-[#1C1612]/50 backdrop-blur-sm"
            onClick={closeMenu}
            aria-hidden="true"
          />

          <motion.div
            ref={dialogRef}
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
            transition={{ duration: 0.25, ease: [0.2, 0.8, 0.2, 1] }}
            className="relative w-full max-w-xs rounded-2xl border border-[#E5DDD4] bg-gradient-to-b from-[#FAF8F5] to-[#F5F1EC] shadow-[0_16px_48px_rgba(28,22,18,0.18)] overflow-hidden"
          >
            <button
              type="button"
              onClick={closeMenu}
              className="close-button absolute top-4 end-4 z-10"
              aria-label={msg("auto.features.tutorial.components.tutorial.menu.literal.1")}
            >
              <X />
            </button>

            <div className="flex flex-col items-center px-6 pt-7 pb-6 text-center">
              <div className="size-12 rounded-xl bg-[#F0EBE4] flex items-center justify-center mb-4">
                <GraduationCap className="size-6 text-[#8C7A6B]" />
              </div>
              <h2 id={titleId} className="text-lg font-bold text-[#3D2E22] mb-1">
                {msg("auto.features.tutorial.components.tutorial.menu.1")}
              </h2>
              <p className="text-xs text-[#8C7A6B] leading-relaxed mb-5">
                {msg("auto.features.tutorial.components.tutorial.menu.2")}
              </p>
              <button
                ref={startBtnRef}
                type="button"
                onClick={() => startTrack("deep-dive")}
                className="w-full px-5 py-2.5 rounded-xl text-sm font-semibold bg-[#3D2E22] text-[#FAF8F5] hover:bg-[#2C2018] transition-colors cursor-pointer"
              >
                {msg("auto.features.tutorial.components.tutorial.menu.3")}
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
