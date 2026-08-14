"use client";

import * as React from "react";
import { createPortal } from "react-dom";
import { useRouter, usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useTutorialContext } from "./tutorial-provider";
import { getLoadedTrack } from "../lib/steps-loader";
import { SpotlightMask } from "./spotlight-mask";
import { TutorialPopover } from "./tutorial-popover";
import { AnimatedWordmark } from "@/shared/ui/animated-wordmark";
import { isTutorialNavigating, registerTutorialHook } from "../lib/bridge";
import { getActiveDir } from "@/shared/lib/runtime-locale";

export function TutorialOverlay() {
  const { state, currentStep, nextStep, prevStep, exitTutorial, completeTrack, toggleAutoPlay } =
    useTutorialContext();
  const pathname = usePathname();

  const [targetRect, setTargetRect] = React.useState<DOMRect | null>(null);
  const [popoverPosition, setPopoverPosition] = React.useState<{
    top: number;
    left: number;
    placement: "top" | "bottom" | "left" | "right";
  } | null>(null);
  const [highlightPadding, setHighlightPadding] = React.useState(8);
  const [highlightRadius, setHighlightRadius] = React.useState(12);
  const [showSplash, setShowSplash] = React.useState(false);
  const [stepReady, setStepReady] = React.useState(false);
  const stepPathRef = React.useRef<string | null>(null);
  const splashTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const router = useRouter();

  // Register splash trigger + client-side navigation with the typed
  // tutorial bridge so steps in lib/tutorial-steps.ts can drive them.
  React.useEffect(() => {
    const unregisterSplash = registerTutorialHook("showTutorialSplash", () => {
      setShowSplash(true);
      // Auto-dismiss: match real submit splash (1.5s) + buffer for navigation.
      // Track via ref so unmount or a second splash cancels the previous timer.
      if (splashTimerRef.current) clearTimeout(splashTimerRef.current);
      splashTimerRef.current = setTimeout(() => {
        splashTimerRef.current = null;
        setShowSplash(false);
      }, 1500);
    });
    const unregisterPush = registerTutorialHook("routerPush", (path: string) => router.push(path));
    return () => {
      unregisterSplash();
      unregisterPush();
      if (splashTimerRef.current) {
        clearTimeout(splashTimerRef.current);
        splashTimerRef.current = null;
      }
    };
  }, [router]);

  const targetRef = React.useRef<Element | null>(null);
  const lastRectRef = React.useRef<{ x: number; y: number; w: number; h: number } | null>(null);
  const autoPlayTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fixed symmetric gap so the popover sits the same distance from the
  // highlighted card regardless of direction — top/bottom use vertical gap,
  // left/right use horizontal gap. Per-step offsetY is reserved only for
  // fine-tuning (e.g. GEPA 2% nudge) and is otherwise 0.
  const FIXED_GAP = 20;
  const calculatePosition = React.useCallback(
    (
      rect: DOMRect,
      placement: "top" | "bottom" | "left" | "right" | "auto",
      opts?: { offsetY?: number; popoverHeight?: number },
    ) => {
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const pw = Math.min(360, vw - 24);
      const mobile = vw < 768;
      const mobileHeightCap = vh * 0.5;
      // Mobile cards scroll internally up to 50dvh. Position against that
      // maximum—not a step's desktop height hint—so a long card never drifts
      // back over its spotlight after the CSS cap takes effect.
      const ph = mobile
        ? Math.min(vh - 24, mobileHeightCap)
        : Math.min(opts?.popoverHeight ?? 260, vh - 24);
      const gap = mobile ? 12 : FIXED_GAP;
      const offsetY = opts?.offsetY ?? 0;

      let p = mobile
        ? vh - rect.bottom >= ph + gap || vh - rect.bottom >= rect.top
          ? ("bottom" as const)
          : ("top" as const)
        : placement;
      if (p === "auto") {
        const spaces = [
          { p: "bottom" as const, s: vh - rect.bottom },
          { p: "top" as const, s: rect.top },
          { p: "right" as const, s: vw - rect.right },
          { p: "left" as const, s: rect.left },
        ];
        p = spaces.sort((a, b) => b.s - a.s)[0]!.p;
      }

      let top = 0,
        left = 0;
      switch (p) {
        case "top":
          top = rect.top - ph - gap + offsetY;
          left = rect.left + rect.width / 2 - pw / 2;
          break;
        case "bottom":
          top = rect.bottom + gap + offsetY;
          left = rect.left + rect.width / 2 - pw / 2;
          break;
        case "left":
          top = rect.top + rect.height / 2 - ph / 2 + offsetY;
          left = rect.left - pw - gap;
          break;
        case "right":
          top = rect.top + rect.height / 2 - ph / 2 + offsetY;
          left = rect.right + gap;
          break;
      }

      top = Math.max(12, Math.min(top, Math.max(12, vh - ph - 12)));
      left = Math.max(12, Math.min(left, Math.max(12, vw - pw - 12)));

      return { top, left, placement: p };
    },
    [],
  );

  const updatePositions = React.useCallback(() => {
    if (!currentStep) return;

    const el = targetRef.current ?? document.querySelector(currentStep.target);
    if (!el) {
      // Keep last-known rect so the spotlight doesn't flash to full-dark
      // mid-transition. Popover is already hidden via stepReady gate.
      return;
    }

    targetRef.current = el;
    const rect = el.getBoundingClientRect();
    // Fixed symmetric highlight for most cards — tight per-step overrides
    // are respected where a target is small (e.g. sidebar nav) to avoid
    // overlapping adjacent tabs.
    const FIXED_HIGHLIGHT_PADDING = 12;
    const FIXED_HIGHLIGHT_RADIUS = 12;
    setHighlightPadding(currentStep.highlightPadding ?? FIXED_HIGHLIGHT_PADDING);
    setHighlightRadius(currentStep.highlightRadius ?? FIXED_HIGHLIGHT_RADIUS);

    // Skip state updates when rect hasn't meaningfully changed — avoids
    // re-renders when an observer fires but nothing moved.
    const prev = lastRectRef.current;
    if (
      prev &&
      Math.abs(prev.x - rect.x) < 0.5 &&
      Math.abs(prev.y - rect.y) < 0.5 &&
      Math.abs(prev.w - rect.width) < 0.5 &&
      Math.abs(prev.h - rect.height) < 0.5
    ) {
      return;
    }
    lastRectRef.current = { x: rect.x, y: rect.y, w: rect.width, h: rect.height };

    setTargetRect(rect);
    setPopoverPosition(
      calculatePosition(rect, currentStep.placement || "auto", {
        offsetY: currentStep.offsetY,
        popoverHeight: currentStep.popoverHeight,
      }),
    );
  }, [currentStep, calculatePosition]);

  React.useEffect(() => {
    if (!state.isVisible || !currentStep) return;
    setStepReady(false);
    // Drop the previous step's rect so the spotlight goes dark for the
    // brief transition rather than animating from the OLD anchor across
    // the screen — moving backward in particular looked broken because
    // the spring kept chasing a stale target while the new step's
    // beforeShow ran.
    setTargetRect(null);
    setPopoverPosition(null);
    lastRectRef.current = null;
    stepPathRef.current = null;
    targetRef.current = null;

    let cancelled = false;
    let waitRaf = 0;
    let trackRaf = 0;
    let resizeObserver: ResizeObserver | null = null;
    const onWindowChange = () => updatePositions();

    const init = async () => {
      if (currentStep.beforeShow) {
        await currentStep.beforeShow();
      }
      if (cancelled) return;

      // Wait for the target to mount (handles route transitions and
      // late-mounting React subtrees). Window must exceed the longest
      // beforeShow waitForElement so steps that navigate to a slow-mounting
      // route (e.g. /optimizations/[id] with demo data) aren't
      // auto-skipped while their anchor is still hydrating.
      const isVisible = (e: Element | null) => {
        if (!e) return false;
        const r = (e as HTMLElement).getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      };
      const el = await new Promise<Element | null>((resolve) => {
        const found = document.querySelector(currentStep.target);
        if (found && isVisible(found)) {
          resolve(found);
          return;
        }
        const start = Date.now();
        const tick = () => {
          if (cancelled) {
            resolve(null);
            return;
          }
          const next = document.querySelector(currentStep.target);
          if ((next && isVisible(next)) || Date.now() - start > 5000) {
            resolve(next && isVisible(next) ? next : null);
            return;
          }
          waitRaf = requestAnimationFrame(tick);
        };
        waitRaf = requestAnimationFrame(tick);
      });
      if (cancelled) return;

      if (!el) {
        // Skip in the SAME direction the user was navigating. Without this,
        // Backspace on a step whose anchor is gone (e.g. wizard remounted)
        // calls nextStep() and races toward COMPLETE_TRACK at the last step,
        // closing the tutorial instead of stepping back to a working anchor.
        const goingBack = state.lastDirection === "backward";
        console.warn(
          `[tutorial] step "${currentStep.id}" target not found: ${currentStep.target} — skipping ${goingBack ? "backward" : "forward"}`,
        );
        if (goingBack) prevStep();
        else nextStep();
        return;
      }

      // Desktop centers compact targets. Mobile aligns them below the app
      // header so the 50dvh tutorial card has room on one side instead of
      // covering the spotlight. scroll-margin works for both the viewport and
      // nested page scrollers without guessing which ancestor owns scrolling.
      const rect = el.getBoundingClientRect();
      const fitsViewport = rect.height <= window.innerHeight - 32;
      const mobile = window.innerWidth < 768;
      if (mobile) {
        const target = el as HTMLElement;
        const previousScrollMarginTop = target.style.scrollMarginTop;
        target.style.scrollMarginTop = "65px";
        target.scrollIntoView({ behavior: "instant" as ScrollBehavior, block: "start" });
        await new Promise((r) => setTimeout(r, 60));
        target.style.scrollMarginTop = previousScrollMarginTop;
        if (cancelled) return;
      } else if (fitsViewport) {
        el.scrollIntoView({ behavior: "instant" as ScrollBehavior, block: "center" });
        await new Promise((r) => setTimeout(r, 60));
        if (cancelled) return;
      } else if (rect.top < 0 || rect.bottom > window.innerHeight) {
        el.scrollIntoView({ behavior: "instant" as ScrollBehavior, block: "start" });
        await new Promise((r) => setTimeout(r, 60));
        if (cancelled) return;
      }

      targetRef.current = el;
      // Two rAFs + a 100ms settle handles route + wizard + AnimatePresence
      // enter transitions where the element exists but is still at 0×0 or
      // mid-slide. Without this the spotlight measured a 0×0 rect at -8,-8
      // when coming backward from the agent to the module picker.
      await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
      await new Promise<void>((r) => setTimeout(r, 120));
      if (cancelled) return;
      // Re-query after settle — tab/wizard remount may have replaced the node
      const fresh = document.querySelector(currentStep.target) as Element | null;
      if (fresh) {
        const fr = (fresh as HTMLElement).getBoundingClientRect();
        if (fr.width > 0 && fr.height > 0) targetRef.current = fresh;
      }
      updatePositions();
      // Observe size changes on the target; scroll/resize cover
      // viewport-driven shifts. The 100ms rAF poll is the safety net for
      // layout shifts that no observer fires for — e.g. surrounding
      // content above the target finishing async loads and pushing it
      // down. updatePositions is a no-op when the rect didn't move
      // ≥0.5px, so the poll is cheap.
      resizeObserver = new ResizeObserver(() => updatePositions());
      resizeObserver.observe(targetRef.current ?? el);
      window.addEventListener("scroll", onWindowChange, { passive: true, capture: true });
      window.addEventListener("resize", onWindowChange);
      let lastTrack = 0;
      const trackTick = (t: number) => {
        if (cancelled) return;
        if (t - lastTrack >= 100) {
          lastTrack = t;
          updatePositions();
        }
        trackRaf = requestAnimationFrame(trackTick);
      };
      trackRaf = requestAnimationFrame(trackTick);
      stepPathRef.current = window.location.pathname;
      setStepReady(true);
    };

    void init();

    return () => {
      cancelled = true;
      if (waitRaf) cancelAnimationFrame(waitRaf);
      if (trackRaf) cancelAnimationFrame(trackRaf);
      if (resizeObserver) resizeObserver.disconnect();
      window.removeEventListener("scroll", onWindowChange, {
        capture: true,
      } as EventListenerOptions);
      window.removeEventListener("resize", onWindowChange);
      // Best-effort per-step cleanup. Closure captures the OLD step, which
      // is what we want — clean up the step we're leaving before the next
      // one's beforeShow runs. Fire-and-forget; UI undo doesn't need await.
      if (currentStep.afterHide) {
        void currentStep.afterHide();
      }
    };
  }, [state.isVisible, state.lastDirection, currentStep, updatePositions, nextStep, prevStep]);

  // Detect manual navigation away from the active step's expected route
  // and exit the tour — the spotlight would otherwise point at a missing
  // element. The user's intentional navigation stands; we don't bounce
  // them back.
  React.useEffect(() => {
    if (!state.isVisible || !stepReady) return;
    if (!stepPathRef.current) return;
    // Check window.location.pathname (truth) instead of pathname
    // (React state from usePathname). The React value can lag behind during
    // route transitions, causing a transient mismatch with stepPathRef
    // (which init() sets from window.location.pathname). pathname stays in
    // deps so the effect still re-runs on every navigation.
    if (window.location.pathname !== stepPathRef.current) {
      exitTutorial();
    }
  }, [pathname, state.isVisible, stepReady, exitTutorial]);

  React.useEffect(() => {
    // Clear any pending timer before deciding whether to arm a new one.
    // The ref makes the "at most one autoplay timer alive" invariant
    // explicit and survives PREV/NEXT/pause races where two effect runs
    // could otherwise overlap if beforeShow resolves slowly.
    if (autoPlayTimerRef.current) {
      clearTimeout(autoPlayTimerRef.current);
      autoPlayTimerRef.current = null;
    }

    if (!stepReady || !state.isAutoPlaying || !currentStep) return;
    const track = state.activeTrack ? (getLoadedTrack(state.activeTrack) ?? null) : null;
    if (!track) return;

    const isLast = state.currentStepIndex >= track.steps.length - 1;
    const duration = (currentStep.readingTimeSec ?? 10) * 1000;

    autoPlayTimerRef.current = setTimeout(() => {
      autoPlayTimerRef.current = null;
      if (isLast) completeTrack();
      else nextStep();
    }, duration);

    return () => {
      if (autoPlayTimerRef.current) {
        clearTimeout(autoPlayTimerRef.current);
        autoPlayTimerRef.current = null;
      }
    };
  }, [
    stepReady,
    state.isAutoPlaying,
    state.activeTrack,
    state.currentStepIndex,
    currentStep,
    nextStep,
    completeTrack,
  ]);

  const handleExit = React.useCallback(() => {
    exitTutorial();
    // Always return to the dashboard so the user never lands on a page
    // still showing fake tutorial data (demo optimization, demo grid,
    // demo grid, etc.). The dashboard clears its demo overlay via
    // the `tutorial-exited` event.
    if (window.location.pathname !== "/") {
      router.push("/");
    }
  }, [exitTutorial, router]);

  React.useEffect(() => {
    if (!state.isVisible) return;

    const onKey = (e: KeyboardEvent) => {
      // Skip when the user is typing into an editable surface — otherwise
      // Enter/Backspace/arrow-keys hijack the tutorial when the user just
      // wants to type into the demo's signature/metric editors or the
      // wizard's name field.
      const tgt = e.target as HTMLElement | null;
      if (tgt) {
        const tag = tgt.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tgt.isContentEditable) {
          return;
        }
      }
      // Physical arrow keys follow reading direction: in LTR, Right advances and
      // Left goes back; they swap in RTL. Enter always advances, Backspace retreats.
      const rtl = getActiveDir() === "rtl";
      const forwardKey = rtl ? "ArrowLeft" : "ArrowRight";
      const backKey = rtl ? "ArrowRight" : "ArrowLeft";
      if (e.key === "Enter" || e.key === forwardKey) {
        e.preventDefault();
        nextStep();
      } else if (e.key === "Backspace" || e.key === backKey) {
        e.preventDefault();
        prevStep();
      } else if (e.key === "Escape") {
        e.preventDefault();
        handleExit();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.isVisible, nextStep, prevStep, handleExit]);

  // Splash must render independently of tutorial visibility
  const splashPortal = showSplash
    ? createPortal(
        <AnimatePresence>
          <motion.div
            className="fixed inset-0 z-[99999] flex items-center justify-center"
            style={{ backgroundColor: "#F0EBE4" }}
            initial={{ y: "-100%" }}
            animate={{ y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            >
              <AnimatedWordmark size={64} autoMorph morphSpeed={120} />
            </motion.div>
          </motion.div>
        </AnimatePresence>,
        document.body,
      )
    : null;

  if (!state.isVisible || !currentStep) return splashPortal;
  if (isTutorialNavigating()) return splashPortal;

  const track = state.activeTrack ? (getLoadedTrack(state.activeTrack) ?? null) : null;
  if (!track) return splashPortal;

  const stepNumber = state.currentStepIndex + 1;
  const totalSteps = Math.max(track.steps.length, stepNumber);
  const isFirst = state.currentStepIndex === 0;
  const isLast = state.currentStepIndex === track.steps.length - 1;

  const handleNext = () => {
    if (isLast) completeTrack();
    else nextStep();
  };

  return (
    <>
      {splashPortal}
      {createPortal(
        <div className="fixed inset-0 z-[9998] pointer-events-none">
          <SpotlightMask
            targetRect={targetRect}
            padding={highlightPadding}
            borderRadius={highlightRadius}
          />

          <AnimatePresence mode="wait">
            {stepReady && popoverPosition && (
              <TutorialPopover
                key={currentStep.id}
                step={currentStep}
                stepNumber={stepNumber}
                totalSteps={totalSteps}
                position={popoverPosition}
                onNext={handleNext}
                onPrev={prevStep}
                onExit={handleExit}
                isFirst={isFirst}
                isLast={isLast}
                isAutoPlaying={state.isAutoPlaying}
                onToggleAutoPlay={toggleAutoPlay}
              />
            )}
          </AnimatePresence>
        </div>,
        document.body,
      )}
    </>
  );
}
