"use client";

import * as React from "react";
import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import { msg } from "@/shared/lib/messages";
import { TokenSourceToggle } from "@/features/billing";
import { DemoBeforeAfter } from "./DemoBeforeAfter";
import { UploadBaseline } from "./UploadBaseline";

const EASE_OUT = [0.16, 1, 0.3, 1] as const;

/** One numbered onboarding moment with a heading, blurb, and its surface. */
function Moment({
  index,
  title,
  description,
  children,
}: {
  index: number;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.section
      initial={reduce ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: EASE_OUT, delay: 0.06 * index }}
      className="flex flex-col gap-3"
    >
      <div className="flex flex-col gap-1">
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        <p className="max-w-[60ch] text-sm leading-relaxed text-muted-foreground">{description}</p>
      </div>
      {children}
    </motion.section>
  );
}

/**
 * Onboarding first-run — the time-delay collapse.
 *
 * Walks a brand-new account from a pre-baked demo before/after (feel the magic
 * in seconds) → a one-tap managed/BYOK choice (managed preselected, no first-win
 * tax) → uploading their own data for an instant baseline framing → into the
 * guaranteed, free first run. No payment wall sits anywhere in this path; the
 * primary affordance — "Optimize, your first run is free" — lives at the end.
 */
export function OnboardingView() {
  const reduce = useReducedMotion();

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-10 px-4 py-10 sm:px-6 sm:py-14">
      <motion.header
        initial={reduce ? false : { opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: EASE_OUT }}
        className="flex flex-col gap-3"
      >
        <span className="text-xs font-semibold uppercase tracking-[0.14em] text-[#8a6d44]">
          {msg("onboarding.eyebrow")}
        </span>
        <h1 className="text-[clamp(1.75rem,4vw,2.5rem)] font-semibold leading-tight text-foreground">
          {msg("onboarding.title")}
        </h1>
        <p className="max-w-[60ch] text-sm leading-relaxed text-muted-foreground">
          {msg("onboarding.lead")}
        </p>
      </motion.header>

      <Moment index={1} title={msg("onboarding.demo.title")} description={msg("onboarding.demo.desc")}>
        <DemoBeforeAfter />
      </Moment>

      <Moment index={2} title={msg("onboarding.mode.title")} description={msg("onboarding.mode.desc")}>
        <TokenSourceToggle />
      </Moment>

      <Moment
        index={3}
        title={msg("onboarding.upload.title")}
        description={msg("onboarding.upload.desc")}
      >
        <UploadBaseline />
      </Moment>

      <Link
        href="/submit"
        className="self-start text-xs font-medium text-muted-foreground underline-offset-4 hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 rounded"
      >
        {msg("onboarding.skip")}
      </Link>
    </div>
  );
}
