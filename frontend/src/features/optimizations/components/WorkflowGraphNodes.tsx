"use client";

/**
 * Node renderers for the read-only workflow graph on the detail page.
 *
 * Deliberately separate from the submit editor's cards: each node kind gets
 * its own accent design (tinted header, colored icon chip, tinted border) so
 * the graph reads at a glance, and no editor state — issues, traces, module
 * settings — is shown. The canvas is always LTR: inputs enter on the left,
 * outputs exit on the right.
 */

import * as React from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

import { ArrowLineRight, Code, Flag, Sparkle, Wrench } from "@/shared/ui/icons";
import { cn } from "@/shared/lib/utils";
import { msg } from "@/shared/lib/messages";
import type { WorkflowNodeSpec } from "@/shared/types/api";
import { nodePorts, type PortInfo } from "@/features/submit/workflow/model";

export interface ViewNodeData extends Record<string, unknown> {
  spec: WorkflowNodeSpec;
}

export type ViewNode = Node<ViewNodeData>;

// Sage input, plum output, golden LLM, slate transform, terracotta tool.
const KIND_STYLES = {
  input: {
    icon: ArrowLineRight,
    border: "border-[#D6DDC6] hover:border-[#5F7A4A]",
    header: "bg-[#F1F4EA]",
    chip: "bg-[#E3EAD6] text-[#5F7A4A]",
    label: "text-[#5F7A4A]",
  },
  output: {
    icon: Flag,
    border: "border-[#E2CFD5] hover:border-[#8A4F5C]",
    header: "bg-[#F6EEF0]",
    chip: "bg-[#F0E0E5] text-[#8A4F5C]",
    label: "text-[#8A4F5C]",
  },
  signature: {
    icon: Sparkle,
    border: "border-[#E5D8BC] hover:border-[#A07C3B]",
    header: "bg-[#F9F3E4]",
    chip: "bg-[#F3E8CF] text-[#A07C3B]",
    label: "text-[#A07C3B]",
  },
  transform: {
    icon: Code,
    border: "border-[#CFDCE4] hover:border-[#4E6E81]",
    header: "bg-[#EEF3F6]",
    chip: "bg-[#E0EAF0] text-[#4E6E81]",
    label: "text-[#4E6E81]",
  },
  mcp: {
    icon: Wrench,
    border: "border-[#E6D2C4] hover:border-[#A65E3F]",
    header: "bg-[#F8EFE9]",
    chip: "bg-[#F4E3D7] text-[#A65E3F]",
    label: "text-[#A65E3F]",
  },
} as const;

function kindLabel(kind: WorkflowNodeSpec["kind"]): string {
  switch (kind) {
    case "input":
      return msg("workflow.node.input");
    case "output":
      return msg("workflow.node.output");
    case "signature":
      return msg("workflow.node.signature");
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

function ViewNodeCard({ data, selected }: NodeProps<ViewNode>) {
  const { spec } = data;
  const style = KIND_STYLES[spec.kind];
  const Icon = style.icon;
  const ports = nodePorts(spec);

  return (
    <div
      className={cn(
        "wf-node-enter min-w-44 max-w-60 select-none rounded-xl border-[1.5px] bg-card text-card-foreground",
        "shadow-[0_1px_3px_rgba(61,46,34,0.08)] transition-[box-shadow,border-color] duration-150",
        selected
          ? "border-[#3D2E22] shadow-[0_4px_16px_rgba(61,46,34,0.16)] ring-4 ring-[#3D2E22]/10"
          : cn(style.border, "hover:shadow-[0_3px_10px_rgba(61,46,34,0.12)]"),
      )}
      data-node-kind={spec.kind}
    >
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-t-[10px] border-b border-border/60 px-3 py-1.5",
          style.header,
        )}
      >
        <span
          className={cn("flex size-5 shrink-0 items-center justify-center rounded-md", style.chip)}
        >
          <Icon className="size-3.5" />
        </span>
        <span className="truncate text-xs font-semibold text-foreground" dir="ltr">
          {displayName(spec)}
        </span>
        <span
          className={cn(
            "ms-auto shrink-0 text-[0.625rem] font-medium uppercase tracking-wide",
            style.label,
          )}
        >
          {kindLabel(spec.kind)}
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
    </div>
  );
}

export const VIEW_NODE_TYPES = {
  input_anchor: ViewNodeCard,
  output_anchor: ViewNodeCard,
  signature: ViewNodeCard,
  transform: ViewNodeCard,
  mcp: ViewNodeCard,
} as const;
