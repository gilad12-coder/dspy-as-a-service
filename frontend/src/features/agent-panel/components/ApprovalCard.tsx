"use client";

import * as React from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { msg } from "@/shared/lib/messages";

import { Button } from "@/shared/ui/primitives/button";
import { cn } from "@/shared/lib/utils";

import { EntryRow } from "./EntryRow";
import { getToolMeta, type ApprovalSeverity } from "../lib/tool-meta";
import type { PendingApprovalPayload } from "../lib/types";

interface ApprovalCardProps {
  payload: PendingApprovalPayload;
  onResolve: (approved: boolean) => void | Promise<void>;
  className?: string;
}

const SEVERITY_STYLES: Record<
  ApprovalSeverity,
  {
    border: string;
    bg: string;
    headerBorder: string;
    iconBg: string;
    iconText: string;
    titleText: string;
    descText: string;
    footerBg: string;
    footerBorder: string;
    confirmVariant: "destructive" | "default";
  }
> = {
  destructive: {
    border: "border-[#9B2C1F]/30",
    bg: "bg-[#FCEFEB]",
    headerBorder: "border-[#9B2C1F]/20",
    iconBg: "bg-[#9B2C1F]/15",
    iconText: "text-[#9B2C1F]",
    titleText: "text-[#7A1E13]",
    descText: "text-[#7A1E13]/80",
    footerBg: "bg-[#F7DED5]/50",
    footerBorder: "border-[#9B2C1F]/15",
    confirmVariant: "destructive",
  },
  warning: {
    border: "border-[#A85A1A]/25",
    bg: "bg-[#FBF3E7]",
    headerBorder: "border-[#A85A1A]/15",
    iconBg: "bg-[#A85A1A]/15",
    iconText: "text-[#A85A1A]",
    titleText: "text-[#7A3E12]",
    descText: "text-[#7A3E12]/80",
    footerBg: "bg-[#F5EAD6]/40",
    footerBorder: "border-[#A85A1A]/10",
    confirmVariant: "default",
  },
  info: {
    border: "border-border/50",
    bg: "bg-muted/40",
    headerBorder: "border-border/40",
    iconBg: "bg-foreground/10",
    iconText: "text-foreground/70",
    titleText: "text-foreground",
    descText: "text-muted-foreground",
    footerBg: "bg-muted/50",
    footerBorder: "border-border/40",
    confirmVariant: "default",
  },
};

/**
 * Approval prompt rendered inline in the chat thread when the agent is
 * about to run a gated MCP tool. Translates the technical tool name and
 * arguments into Hebrew plain-language copy, with the raw technical
 * details tucked behind a disclosure toggle.
 */
export function ApprovalCard({ payload, onResolve, className }: ApprovalCardProps) {
  const [busy, setBusy] = React.useState<"approve" | "deny" | null>(null);
  const [showDetails, setShowDetails] = React.useState(false);

  const handle = async (approved: boolean) => {
    setBusy(approved ? "approve" : "deny");
    try {
      await onResolve(approved);
    } finally {
      setBusy(null);
    }
  };

  const meta = getToolMeta(payload.tool);
  const styles = SEVERITY_STYLES[meta.severity];
  const Icon = meta.icon;
  const entries = Object.entries(payload.arguments ?? {});
  const hasArgs = entries.length > 0;

  return (
    <div
      className={cn(
        "rounded-2xl border shadow-sm overflow-hidden",
        styles.border,
        styles.bg,
        className,
      )}
    >
      <div className={cn("flex items-start gap-2.5 px-4 py-3 border-b", styles.headerBorder)}>
        <span
          className={cn(
            "inline-flex size-8 shrink-0 items-center justify-center rounded-full",
            styles.iconBg,
            styles.iconText,
          )}
        >
          <Icon className="size-4" aria-hidden="true" />
        </span>
        <div className="min-w-0 flex-1">
          <div className={cn("text-[0.875rem] font-semibold leading-tight", styles.titleText)}>
            {meta.title}
          </div>
          <div className={cn("text-[0.75rem] mt-1 leading-snug", styles.descText)}>
            {meta.description}
          </div>
        </div>
      </div>

      {hasArgs && (
        <div className="px-4 py-2">
          <button
            type="button"
            onClick={() => setShowDetails((v) => !v)}
            className={cn(
              "inline-flex items-center gap-1 text-[0.6875rem] transition-opacity hover:opacity-75 cursor-pointer",
              styles.descText,
            )}
            aria-expanded={showDetails}
          >
            <ChevronDown
              className={cn("size-3 transition-transform", showDetails && "rotate-180")}
              aria-hidden="true"
            />
            {showDetails
              ? msg("auto.features.agent.panel.components.approvalcard.literal.1")
              : msg("auto.features.agent.panel.components.approvalcard.literal.2")}
          </button>
          {showDetails && (
            <dl className="mt-2 space-y-2 text-[0.75rem]">
              {entries.map(([key, value]) => (
                <EntryRow key={key} argKey={key} value={value} labelClassName={styles.descText} />
              ))}
            </dl>
          )}
        </div>
      )}

      <div
        className={cn(
          "flex items-center gap-2 px-4 py-2.5 border-t",
          styles.footerBg,
          styles.footerBorder,
        )}
      >
        <Button
          variant="outline"
          onClick={() => handle(false)}
          disabled={busy !== null}
          className="flex-1 justify-center"
        >
          {busy === "deny" ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            msg("auto.features.agent.panel.components.approvalcard.literal.3")
          )}
        </Button>
        <Button
          variant={styles.confirmVariant}
          onClick={() => handle(true)}
          disabled={busy !== null}
          className="flex-1 justify-center"
        >
          {busy === "approve" ? <Loader2 className="size-4 animate-spin" /> : meta.confirmLabel}
        </Button>
      </div>
    </div>
  );
}
