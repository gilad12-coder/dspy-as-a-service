"use client";

/**
 * Read-only rendering of a run's workflow graph for the detail page.
 *
 * Interactive but never mutating: zoom/pan with on-canvas controls, a
 * fullscreen mode (portal pattern shared with the submit editor canvas),
 * and a click-to-inspect panel showing each node's spec — signature code,
 * transform code, ports, tools — without any editing affordances.
 */

import * as React from "react";
import { createPortal } from "react-dom";
import dynamic from "next/dynamic";
import {
  Background,
  MarkerType,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useStore,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { AnimatePresence, motion } from "framer-motion";

import {
  ArrowCounterClockwise,
  ArrowsIn,
  ArrowsOut,
  MagnifyingGlassMinus,
  MagnifyingGlassPlus,
  X,
} from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { Label } from "@/shared/ui/primitives/label";
import { Skeleton } from "@/shared/ui/skeleton";
import { cn } from "@/shared/lib/utils";
import { msg } from "@/shared/lib/messages";
import { autoLayoutSpec } from "@/features/submit/workflow/model";
import { NODE_TYPES, flowTypeFor, type CanvasNode } from "@/features/submit/workflow/nodes";
import type { WorkflowNodeSpec, WorkflowSpec } from "@/shared/types/api";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={160} borderRadius={8} />,
});

const FIT_VIEW = { padding: 0.2, maxZoom: 1 };

export function WorkflowGraphView({ spec }: { spec: WorkflowSpec }) {
  return (
    <ReactFlowProvider>
      <GraphView spec={spec} />
    </ReactFlowProvider>
  );
}

function GraphView({ spec }: { spec: WorkflowSpec }) {
  const { fitView } = useReactFlow();
  const [fullscreen, setFullscreen] = React.useState(false);
  const [inspectorId, setInspectorId] = React.useState<string | null>(null);

  // Runs submitted through the canvas carry positions; auto-layout covers
  // programmatically submitted specs that ship without them.
  const laid = React.useMemo(
    () => (spec.nodes.every((n) => n.position) ? spec : autoLayoutSpec(spec)),
    [spec],
  );
  const nodes: CanvasNode[] = React.useMemo(
    () =>
      laid.nodes.map((node) => ({
        id: node.id,
        type: flowTypeFor(node),
        position: node.position ?? { x: 0, y: 0 },
        selected: node.id === inspectorId,
        data: { spec: node, issues: [], trace: null, pulse: false },
      })),
    [laid, inspectorId],
  );
  const edges: Edge[] = React.useMemo(
    () =>
      laid.edges.map((e) => ({
        id: `${e.source}.${e.source_port}->${e.target}.${e.target_port}`,
        source: e.source,
        target: e.target,
        sourceHandle: e.source_port,
        targetHandle: e.target_port,
        style: { stroke: "#8A7563", strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#8A7563", width: 16, height: 16 },
      })),
    [laid],
  );

  // ESC closes the details panel first, then exits fullscreen — same modal
  // stack as the editor canvas.
  React.useEffect(() => {
    if (!fullscreen && !inspectorId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      if (inspectorId) setInspectorId(null);
      else setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen, inspectorId]);

  // Lock page scroll while the fullscreen overlay is open.
  React.useEffect(() => {
    if (!fullscreen) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prevOverflow;
    };
  }, [fullscreen]);

  // Re-fit after the portal moves the canvas between containers.
  React.useEffect(() => {
    const t = window.setTimeout(() => fitView({ ...FIT_VIEW, duration: 200 }), 80);
    return () => window.clearTimeout(t);
  }, [fullscreen, fitView]);

  const inspectorNode = laid.nodes.find((n) => n.id === inspectorId) ?? null;
  // The details panel slides in from the inline end; page direction decides
  // which physical side that is. Client-only component (ssr: false import).
  const slideFrom = document.documentElement.dir === "rtl" ? -24 : 24;

  const body = (
    <div
      className={cn(
        fullscreen
          ? "fixed inset-0 z-50 flex h-screen w-screen flex-col bg-background"
          : "relative flex flex-col overflow-hidden rounded-lg border border-border/60",
      )}
    >
      <div className="relative flex min-h-0 flex-1">
        <div dir="ltr" className={cn("min-w-0 flex-1", fullscreen ? "min-h-0" : "h-[480px]")}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodeClick={(_, node) => setInspectorId(node.id)}
            onPaneClick={() => setInspectorId(null)}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            edgesFocusable={false}
            zoomOnDoubleClick={false}
            minZoom={0.1}
            maxZoom={3}
            fitView
            fitViewOptions={FIT_VIEW}
            proOptions={{ hideAttribution: true }}
            className="bg-[#FDFCFA]"
          >
            <Background gap={16} size={1.25} color="#E3D9CB" />
            <ViewControls />
            <Panel position="top-right" className="!m-3">
              <div className="flex items-center overflow-hidden rounded-lg border border-border/60 bg-background/95 shadow-sm backdrop-blur">
                <ControlButton
                  icon={fullscreen ? ArrowsIn : ArrowsOut}
                  label={msg(
                    fullscreen
                      ? "workflow.toolbar.exit_fullscreen"
                      : "workflow.toolbar.fullscreen",
                  )}
                  onClick={() => setFullscreen((f) => !f)}
                />
              </div>
            </Panel>
            <Panel position="bottom-center" className="pointer-events-none !m-3">
              <span
                dir="auto"
                className="whitespace-nowrap rounded-full border border-border/50 bg-background/80 px-3 py-1 text-[0.6875rem] text-muted-foreground/80 shadow-xs backdrop-blur"
              >
                {msg("optimization.workflow.hint")}
              </span>
            </Panel>
          </ReactFlow>
        </div>
        <AnimatePresence>
          {inspectorNode && (
            <motion.div
              key="node-details"
              initial={{ x: slideFrom, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: slideFrom, opacity: 0 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
              className={cn(
                "absolute inset-y-0 end-0 z-20 overflow-hidden border-s border-border/40 bg-card shadow-xl",
                fullscreen ? "w-[min(360px,88vw)]" : "w-[min(320px,88vw)]",
              )}
            >
              <NodeDetails spec={inspectorNode} onClose={() => setInspectorId(null)} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );

  if (!fullscreen) return body;
  return (
    <>
      {/* Placeholder keeps the tab's height while the canvas lives in the
          portal, so the page doesn't collapse behind the overlay. */}
      <div className="h-[480px] rounded-lg border border-border/60 bg-muted/20" aria-hidden />
      {createPortal(body, document.body)}
    </>
  );
}

function ViewControls() {
  const { zoomIn, zoomOut, zoomTo, fitView } = useReactFlow();
  const zoom = useStore((s) => s.transform[2]);
  return (
    <Panel position="bottom-left" className="!m-3">
      <div className="flex items-center overflow-hidden rounded-lg border border-border/60 bg-background/95 shadow-sm backdrop-blur">
        <ControlButton
          icon={MagnifyingGlassMinus}
          label={msg("workflow.controls.zoom_out")}
          onClick={() => zoomOut({ duration: 150 })}
        />
        <button
          type="button"
          onClick={() => zoomTo(1, { duration: 200 })}
          title={msg("workflow.controls.zoom_reset")}
          className="h-7 min-w-12 cursor-pointer px-1 text-center text-[0.6875rem] font-medium tabular-nums text-muted-foreground transition-colors hover:text-foreground"
        >
          {Math.round(zoom * 100)}%
        </button>
        <ControlButton
          icon={MagnifyingGlassPlus}
          label={msg("workflow.controls.zoom_in")}
          onClick={() => zoomIn({ duration: 150 })}
        />
        <div className="h-4 w-px bg-border/70" />
        <ControlButton
          icon={ArrowCounterClockwise}
          label={msg("workflow.controls.reset")}
          onClick={() => fitView({ ...FIT_VIEW, duration: 300 })}
        />
      </div>
    </Panel>
  );
}

function ControlButton({
  icon: Icon,
  label,
  onClick,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className="flex h-7 w-7 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
    >
      <Icon className="size-3.5" />
    </button>
  );
}

const codeHeight = (code: string): string =>
  `${Math.min((code.split("\n").length + 1) * 19.6 + 8, 340)}px`;

/** Read-only details for the clicked node, mirroring the editor inspector's layout. */
function NodeDetails({ spec, onClose }: { spec: WorkflowNodeSpec; onClose: () => void }) {
  return (
    <div className="flex h-full flex-col overflow-y-auto bg-card">
      <div className="flex items-center justify-between gap-2 border-b border-border/60 px-4 py-2.5">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold" dir="ltr">
            {spec.name ?? spec.id}
          </div>
          <div className="text-[0.6875rem] text-muted-foreground">
            {msg(`workflow.inspector.kind.${spec.kind}`)}
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          className="shrink-0 text-muted-foreground hover:text-foreground"
          aria-label={msg("workflow.inspector.close")}
        >
          <X className="size-3.5" />
        </Button>
      </div>

      <div className="space-y-4 px-4 py-3">
        {(spec.kind === "input" || spec.kind === "output") && (
          <FieldList
            label={msg(
              spec.kind === "input"
                ? "workflow.inspector.input_fields"
                : "workflow.inspector.output_fields",
            )}
            names={spec.fields.map((f) => f.name)}
          />
        )}

        {spec.kind === "signature" && (
          <>
            {spec.tool_filter && spec.tool_filter.length > 0 && (
              <FieldList label={msg("workflow.inspector.flex_tools")} names={spec.tool_filter} />
            )}
            <Section label={msg("workflow.inspector.signature_code")}>
              <CodeEditor
                value={spec.signature_code}
                onChange={() => {}}
                height={codeHeight(spec.signature_code)}
                readOnly
              />
            </Section>
          </>
        )}

        {spec.kind === "transform" && (
          <>
            <FieldList
              label={msg("workflow.inspector.input_fields")}
              names={spec.input_fields.map((f) => f.name)}
            />
            <FieldList
              label={msg("workflow.inspector.output_fields")}
              names={spec.output_fields.map((f) => f.name)}
            />
            <Section label={msg("workflow.inspector.transform_code")}>
              <CodeEditor
                value={spec.transform_code}
                onChange={() => {}}
                height={codeHeight(spec.transform_code)}
                readOnly
              />
            </Section>
          </>
        )}

        {spec.kind === "mcp" && (
          <>
            <Section label={msg("workflow.inspector.tool_name")}>
              <div className="rounded-md bg-muted px-2 py-1 font-mono text-xs" dir="ltr">
                {spec.tool_name}
              </div>
            </Section>
            <FieldList
              label={msg("workflow.inspector.input_fields")}
              names={spec.input_fields.map((f) => f.name)}
            />
            <FieldList
              label={msg("workflow.inspector.result_field")}
              names={[spec.output_field.name]}
            />
          </>
        )}
      </div>
    </div>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-[0.6875rem] uppercase tracking-wide text-muted-foreground">
        {label}
      </Label>
      {children}
    </div>
  );
}

function FieldList({ label, names }: { label: string; names: string[] }) {
  return (
    <Section label={label}>
      <div className="flex flex-wrap gap-1">
        {names.map((name) => (
          <span
            key={name}
            dir="ltr"
            className="rounded-md bg-muted px-2 py-0.5 font-mono text-xs text-foreground"
          >
            {name}
          </span>
        ))}
      </div>
    </Section>
  );
}

export default WorkflowGraphView;
