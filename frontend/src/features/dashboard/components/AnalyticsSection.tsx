"use client";

import { AnimatePresence, motion } from "framer-motion";
import { CaretDown } from "@/shared/ui/icons";
import { useState, type ReactNode } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";

interface AnalyticsSectionProps {
  title: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
  className?: string;
}

export function AnalyticsSection({
  title,
  defaultOpen = true,
  children,
  className,
}: AnalyticsSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <Card className={className}>
      <CardHeader
        role="button"
        tabIndex={0}
        aria-expanded={isOpen}
        className="pb-3 cursor-pointer select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 rounded-md"
        onClick={() => setIsOpen(!isOpen)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setIsOpen((open) => !open);
          }
        }}
      >
        <CardTitle className="text-base font-semibold flex items-center justify-between">
          <span className="min-w-0">{title}</span>
          <motion.div
            className="shrink-0"
            animate={{ rotate: isOpen ? 0 : 180 }}
            transition={{ duration: 0.3, ease: [0.2, 0.8, 0.2, 1] }}
          >
            <CaretDown className="size-4 text-muted-foreground" />
          </motion.div>
        </CardTitle>
      </CardHeader>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{
              height: { type: "spring", stiffness: 300, damping: 30 },
              opacity: { duration: 0.2 },
            }}
            style={{ overflow: "hidden" }}
          >
            <CardContent className="pt-0">{children}</CardContent>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}
