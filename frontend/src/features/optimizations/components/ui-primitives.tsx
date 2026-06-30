"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Brain, Check, ChevronDown, Clipboard } from "lucide-react";
import { msg } from "@/shared/lib/messages";

export function InfoCard({
  label,
  value,
  icon,
}: {
  label: ReactNode;
  value: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <motion.div
      whileHover={{ y: -1 }}
      transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
      className="group relative rounded-lg border border-[#E3DCD0] bg-[#FBF9F4] px-3.5 py-3 transition-[border-color,box-shadow] duration-200 hover:border-[#C8A882]/55 hover:shadow-[0_2px_8px_-2px_rgba(124,99,80,0.1)]"
    >
      <div className="flex items-center gap-1.5 mb-1.5">
        {icon && (
          <span
            className="shrink-0 inline-flex items-center justify-center size-3.5 text-[#A89680] transition-colors duration-200 group-hover:text-[#7C6350]"
            aria-hidden="true"
          >
            {icon}
          </span>
        )}
        <p className="text-[0.625rem] font-semibold tracking-[0.08em] uppercase text-[#A89680] truncate">
          {label}
        </p>
      </div>
      <p className="text-sm font-semibold text-[#1C1612] truncate">
        {value ?? <span className="text-[#BFB3A3] font-normal">—</span>}
      </p>
    </motion.div>
  );
}

export function LangPicker<T extends string>({
  value,
  onChange,
  labels,
}: {
  value: T;
  onChange: (v: T) => void;
  labels: Record<T, string>;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onClick);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);
  const keys = Object.keys(labels) as T[];
  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 px-1.5 py-0.5 -mx-1.5 -my-0.5 rounded-md font-semibold text-[#7C6350] tracking-wide hover:bg-black/5 transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/50"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span>{labels[value]}</span>
        <ChevronDown
          className={`size-3 text-[#8C7A6B] transition-transform duration-150 ${open ? "rotate-180" : ""}`}
        />
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            role="listbox"
            initial={{ opacity: 0, y: 4, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.96 }}
            transition={{ duration: 0.12, ease: [0.16, 1, 0.3, 1] }}
            className="absolute bottom-full mb-1.5 start-0 z-20 min-w-[120px] max-w-[min(220px,92vw)] rounded-lg border border-[#E5DDD4] bg-[#FAF6F0] shadow-lg overflow-hidden py-1"
          >
            {keys.map((k) => (
              <li key={k}>
                <button
                  type="button"
                  onClick={() => {
                    onChange(k);
                    setOpen(false);
                  }}
                  className={`w-full text-start px-3 py-1.5 text-[0.6875rem] font-semibold tracking-wide transition-colors cursor-pointer flex items-center justify-between ${k === value ? "bg-[#3D2E22]/8 text-[#3D2E22]" : "text-[#7C6350] hover:bg-black/5"}`}
                  role="option"
                  aria-selected={k === value}
                >
                  <span>{labels[k]}</span>
                  {k === value && <Check className="size-3" />}
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}

const REASONING_EFFORT_LABELS: Record<string, string> = {
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
};

function reasoningEffortLabel(value: string | null | undefined): string | null {
  if (!value) return null;
  return REASONING_EFFORT_LABELS[value.toLowerCase()] ?? value;
}

export function ReasoningPill({
  value,
  size = "xs",
}: {
  value: string | null | undefined;
  size?: "xs" | "sm";
}) {
  const label = reasoningEffortLabel(value);
  if (!label) return null;
  const sizing =
    size === "sm" ? "gap-1 px-1.5 py-0.5 text-[10px]" : "gap-0.5 px-1 py-0.5 text-[9px]";
  const iconSize = size === "sm" ? "size-3" : "size-2.5";
  return (
    <span
      className={`shrink-0 inline-flex items-center rounded bg-muted/50 font-semibold text-muted-foreground/80 ${sizing}`}
      title={`Reasoning effort: ${label}`}
    >
      <Brain className={iconSize} />
      {label}
    </span>
  );
}

export function CopyButton({ text, className = "" }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        void navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className={`p-1.5 cursor-pointer transition-opacity duration-200 outline-none border-none shadow-none ring-0 bg-transparent hover:opacity-100 ${className}`}
      title={msg("auto.features.optimizations.components.ui.primitives.literal.1")}
      aria-label={msg("auto.features.optimizations.components.ui.primitives.literal.2")}
    >
      {copied ? (
        <Check className="size-3.5 text-foreground/70" />
      ) : (
        <Clipboard className="size-3.5 text-foreground/40 hover:text-foreground/70" />
      )}
    </button>
  );
}
