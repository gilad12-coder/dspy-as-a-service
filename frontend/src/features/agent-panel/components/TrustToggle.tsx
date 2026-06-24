"use client";

import * as React from "react";
import { Shield, ShieldCheck, Zap } from "lucide-react";
import { formatMsg, msg } from "@/shared/lib/messages";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { cn } from "@/shared/lib/utils";

import { TRUST_MODE_DESCRIPTION, TRUST_MODE_HUE, TRUST_MODE_LABEL } from "../hooks/use-trust-mode";
import type { TrustMode } from "../lib/types";

interface TrustToggleProps {
  mode: TrustMode;
  onCycle: () => void;
  className?: string;
}

const ICONS: Record<TrustMode, React.ComponentType<{ className?: string }>> = {
  ask: Shield,
  auto_safe: ShieldCheck,
  yolo: Zap,
};

const MODE_ORDER: TrustMode[] = ["ask", "auto_safe", "yolo"];

export function TrustToggle({ mode, onCycle, className }: TrustToggleProps) {
  const Icon = ICONS[mode];
  const hue = TRUST_MODE_HUE[mode];
  const label = TRUST_MODE_LABEL[mode];

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onCycle}
          aria-label={formatMsg("auto.features.agent.panel.components.trusttoggle.template.1", {
            p1: label,
          })}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[0.6875rem]",
            "transition-all duration-150 hover:bg-accent/60 active:scale-[0.97] cursor-pointer",
            className,
          )}
          style={{
            borderColor: `${hue}30`,
            color: hue,
            backgroundColor: `${hue}0A`,
          }}
        >
          <Icon className="size-3" aria-hidden="true" />
          <span className="font-medium leading-none">{label}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-[240px] px-3 py-2">
        <div className="font-medium">
          {msg("auto.features.agent.panel.components.trusttoggle.1")}
        </div>
        <ul className="mt-1.5 space-y-1">
          {MODE_ORDER.map((m) => {
            const ModeIcon = ICONS[m];
            const active = m === mode;
            return (
              <li
                key={m}
                className={cn(
                  "flex items-start gap-1.5 leading-tight",
                  active ? "opacity-100" : "opacity-60",
                )}
              >
                <ModeIcon className="size-3 shrink-0 mt-[2px]" aria-hidden="true" />
                <span>
                  <span className={active ? "font-semibold" : "font-medium"}>
                    {TRUST_MODE_LABEL[m]}
                  </span>
                  <span className="opacity-80"> — {TRUST_MODE_DESCRIPTION[m]}</span>
                </span>
              </li>
            );
          })}
        </ul>
      </TooltipContent>
    </Tooltip>
  );
}
