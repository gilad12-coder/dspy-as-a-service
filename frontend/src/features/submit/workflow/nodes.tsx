"use client";

/**
 * Custom React Flow node renderers for the workflow canvas.
 *
 * One shared card shell renders all five node kinds; ports come from
 * `nodePorts(spec)` so signature nodes re-derive their handles live as the
 * user edits code in the inspector. The canvas is always LTR (inputs enter
 * on the left, outputs exit on the right) regardless of page direction.
 */

import * as React from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import {
  ArrowLineRight,
  Brain,
  Check,
  Code,
  FadersHorizontal,
  Flag,
  Lightning,
  Sparkle,
  Warning,
  Wrench,
  X,
} from "@/shared/ui/icons";

import { cn } from "@/shared/lib/utils";
import { msg } from "@/shared/lib/messages";
import type { WorkflowNodeSpec } from "@/shared/types/api";
import { nodePorts, type PortInfo } from "./model";

/** Per-node dry-run replay state, painted onto the card after a run. */
export interface NodeTraceState {
  status: "ok" | "error";
  elapsedMs: number;
  error?: string | null;
}

export interface WorkflowNodeData extends Record<string, unknown> {
  spec: WorkflowNodeSpec;
  issues: string[];
  trace?: NodeTraceState | null;
  // Briefly true right after the code agent changed this node.
  pulse?: boolean;
}

export type CanvasNode = Node<WorkflowNodeData>;

const KIND_ICONS = {
  input: ArrowLineRight,
  output: Flag,
  signature: Sparkle,
  transform: Code,
  mcp: Wrench,
} as const;

// Each LLM module reads as its own building block — distinct icon and accent
// with an "LLM-<module>" badge, so the card needs no separate module caption.
const MODULE_STYLES = {
  predict: {
    icon: Sparkle,
    label: "LLM-Predict",
    header: "bg-[#F9F3E4]",
    accent: "text-[#A07C3B]",
    border: "border-[#E5D8BC] hover:border-[#A07C3B]",
  },
  cot: {
    icon: Brain,
    label: "LLM-CoT",
    header: "bg-[#EDF1F8]",
    accent: "text-[#4A6FA5]",
    border: "border-[#CBD7E9] hover:border-[#4A6FA5]",
  },
  react: {
    icon: Lightning,
    label: "LLM-ReAct",
    header: "bg-[#EAF3F1]",
    accent: "text-[#2F7A6E]",
    border: "border-[#C4DDD7] hover:border-[#2F7A6E]",
  },
  flex: {
    icon: FadersHorizontal,
    label: "LLM-Flex",
    header: "bg-[#F1EEF7]",
    accent: "text-[#7B5EA7]",
    border: "border-[#D7CDE8] hover:border-[#7B5EA7]",
  },
} as const;

function kindLabel(spec: WorkflowNodeSpec): string {
  switch (spec.kind) {
    case "input":
      return msg("workflow.node.input");
    case "output":
      return msg("workflow.node.output");
    case "signature":
      return MODULE_STYLES[spec.module_name].label;
    case "transform":
      return msg("workflow.node.transform");
    case "mcp":
      return msg("workflow.node.mcp");
  }
}

function displayName(spec: WorkflowNodeSpec): string {
  if (spec.name) return spec.name;
  if (spec.kind === "mcp" && spec.tool_name) return spec.tool_name;
  return spec.id;
}

function PortRow({ port, side, nodeId }: { port: PortInfo; side: "in" | "out"; nodeId: string }) {
  const isStr = port.annotation === "str" || port.annotation === "";
  return (
    <div
      className={cn(
        "relative flex items-center px-3 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground",
        side === "in" ? "justify-start" : "justify-end",
      )}
    >
      <span className="truncate font-mono" title={`${port.name}: ${port.annotation || "str"}`}>
        {port.name}
      </span>
      <Handle
        type={side === "in" ? "target" : "source"}
        position={side === "in" ? Position.Left : Position.Right}
        id={port.name}
        className={cn(
          "!size-3 !border-2 !border-[#FDFCFA]",
          isStr ? "!bg-[#3D2E22]" : "!bg-[#C8A882]",
        )}
        aria-label={`${nodeId}.${port.name}`}
      />
    </div>
  );
}

function NodeCard({ data, selected }: NodeProps<CanvasNode>) {
  const { spec, issues, trace, pulse } = data;
  const moduleStyle = spec.kind === "signature" ? MODULE_STYLES[spec.module_name] : null;
  const Icon = moduleStyle?.icon ?? KIND_ICONS[spec.kind];
  const ports = nodePorts(spec);
  const isAnchor = spec.kind === "input" || spec.kind === "output";
  const hasIssues = issues.length > 0;

  return (
    <div
      className={cn(
        "wf-node-enter min-w-44 max-w-60 select-none rounded-xl border-[1.5px] bg-card text-card-foreground",
        "shadow-[0_1px_3px_rgba(61,46,34,0.08)] transition-[box-shadow,border-color] duration-150",
        selected
          ? "border-[#3D2E22] shadow-[0_4px_16px_rgba(61,46,34,0.16)] ring-4 ring-[#3D2E22]/10"
          : cn(
              moduleStyle?.border ?? "border-[#E2D8CA] hover:border-[#C8A882]",
              "hover:shadow-[0_3px_10px_rgba(61,46,34,0.12)]",
            ),
        trace?.status === "error" && "border-destructive",
        pulse && "animate-pulse border-[#C8A882] ring-4 ring-[#C8A882]/40",
      )}
      data-node-kind={spec.kind}
    >
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-t-[10px] border-b border-border/60 px-3 py-1.5",
          moduleStyle?.header ?? (isAnchor ? "bg-[#F3EDE3]" : "bg-[#FAF8F5]"),
        )}
      >
        <Icon className={cn("size-3.5 shrink-0", moduleStyle?.accent ?? "text-[#3D2E22]")} />
        <span className="truncate text-xs font-semibold text-foreground" dir="ltr">
          {displayName(spec)}
        </span>
        <span
          className={cn(
            "ms-auto shrink-0 text-[0.625rem] font-medium uppercase tracking-wide",
            moduleStyle?.accent ?? "text-muted-foreground",
          )}
        >
          {kindLabel(spec)}
        </span>
      </div>
      <div className="grid grid-cols-2 py-1.5">
        <div className="min-w-0">
          {ports.inputs.map((p) => (
            <PortRow key={p.name} port={p} side="in" nodeId={spec.id} />
          ))}
        </div>
        <div className="min-w-0">
          {ports.outputs.map((p) => (
            <PortRow key={p.name} port={p} side="out" nodeId={spec.id} />
          ))}
        </div>
      </div>
      {(hasIssues || trace) && (
        <div className="flex items-center gap-2 border-t border-border/60 px-3 py-1">
          {hasIssues && (
            <span
              className="inline-flex items-center gap-1 text-[0.625rem] font-medium text-[#A3512B]"
              title={issues.join("\n")}
            >
              <Warning className="size-3" />
              {issues.length}
            </span>
          )}
          {trace && (
            <span
              className={cn(
                "ms-auto inline-flex items-center gap-1 text-[0.625rem] font-medium tabular-nums",
                trace.status === "ok" ? "text-[#5A7247]" : "text-destructive",
              )}
              title={trace.error ?? undefined}
            >
              {trace.status === "ok" ? <Check className="size-3" /> : <X className="size-3" />}
              {Math.round(trace.elapsedMs)}
              <span>{msg("workflow.trace.ms")}</span>
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// All kinds share the card; React Flow just needs a type→component map.
export const NODE_TYPES = {
  input_anchor: NodeCard,
  output_anchor: NodeCard,
  signature: NodeCard,
  transform: NodeCard,
  mcp: NodeCard,
} as const;

/** Map a spec node kind to its React Flow node type key. */
export function flowTypeFor(spec: WorkflowNodeSpec): keyof typeof NODE_TYPES {
  if (spec.kind === "input") return "input_anchor";
  if (spec.kind === "output") return "output_anchor";
  return spec.kind;
}
