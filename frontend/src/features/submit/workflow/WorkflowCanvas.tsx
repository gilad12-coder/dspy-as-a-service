"use client";

/**
 * The workflow canvas: an n8n-style node editor over the wizard's
 * `WorkflowSpec`.
 *
 * The spec is the single source of truth (owned by the wizard hook); React
 * Flow state is re-derived from it after every structural mutation. Pure
 * drag movement stays inside React Flow until drag-stop, when positions are
 * committed back onto the spec. External spec replacements (dataset init,
 * clone hydration, future agent ops) bump `specRevision`, which remounts the
 * flow so stale internal state can't survive.
 *
 * The canvas itself is always LTR — data flows left→right — while inspector
 * chrome follows the page direction.
 */

import * as React from "react";
import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type IsValidConnection,
  type NodeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  AlertTriangle,
  Code2,
  LayoutGrid,
  Maximize2,
  Minimize2,
  Play,
  Sparkles,
  Wrench,
  X,
} from "lucide-react";

import { Button } from "@/shared/ui/primitives/button";
import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import type {
  WorkflowDryRunResponse,
  WorkflowNodeSpec,
  WorkflowSpec,
} from "@/shared/types/api";

import {
  nodePorts,
  newNodeSpec,
  validateWorkflowSpec,
  wouldCreateCycle,
  type WorkflowIssue,
} from "./model";
import { workflowIssueText } from "./issue-text";
import { NODE_TYPES, flowTypeFor, type CanvasNode, type NodeTraceState } from "./nodes";
import { NodeInspector } from "./NodeInspector";
import { DryRunDialog } from "./DryRunDialog";

export interface WorkflowDryRunBinding {
  /** Reason the dry-run button is disabled, or null when it can run. */
  disabledReason: string | null;
  /** Prefill values for the input anchor's fields (first dataset row). */
  sampleInputs: Record<string, string>;
  run: (inputs: Record<string, unknown>) => Promise<WorkflowDryRunResponse>;
}

interface WorkflowCanvasProps {
  spec: WorkflowSpec;
  specRevision: number;
  onSpecChange: (spec: WorkflowSpec) => void;
  dryRun?: WorkflowDryRunBinding;
  className?: string;
}

const edgeId = (e: { source: string; source_port: string; target: string; target_port: string }) =>
  `${e.source}.${e.source_port}->${e.target}.${e.target_port}`;

function deriveNodes(
  spec: WorkflowSpec,
  issuesByNode: Map<string, string[]>,
  traces: Map<string, NodeTraceState>,
  prev: CanvasNode[],
): CanvasNode[] {
  const prevById = new Map(prev.map((n) => [n.id, n]));
  return spec.nodes.map((node, index) => ({
    id: node.id,
    type: flowTypeFor(node),
    position: node.position ?? prevById.get(node.id)?.position ?? { x: 80 + index * 60, y: 80 },
    selected: prevById.get(node.id)?.selected ?? false,
    deletable: node.kind !== "input" && node.kind !== "output",
    data: {
      spec: node,
      issues: issuesByNode.get(node.id) ?? [],
      trace: traces.get(node.id) ?? null,
    },
  }));
}

function deriveEdges(spec: WorkflowSpec, prev: Edge[]): Edge[] {
  const prevById = new Map(prev.map((e) => [e.id, e]));
  return spec.edges.map((e) => ({
    id: edgeId(e),
    source: e.source,
    sourceHandle: e.source_port,
    target: e.target,
    targetHandle: e.target_port,
    selected: prevById.get(edgeId(e))?.selected ?? false,
    style: { stroke: "#8A7563", strokeWidth: 1.5 },
  }));
}

function CanvasInner({ spec, onSpecChange, dryRun, className }: Omit<WorkflowCanvasProps, "specRevision">) {
  const { fitView } = useReactFlow();
  const [fullscreen, setFullscreen] = React.useState(false);
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [dryRunOpen, setDryRunOpen] = React.useState(false);
  const [dryRunResult, setDryRunResult] = React.useState<WorkflowDryRunResponse | null>(null);

  const issues: WorkflowIssue[] = React.useMemo(
    () => validateWorkflowSpec(spec, workflowIssueText),
    [spec],
  );
  const issuesByNode = React.useMemo(() => {
    const map = new Map<string, string[]>();
    for (const issue of issues) {
      if (!issue.nodeId) continue;
      map.set(issue.nodeId, [...(map.get(issue.nodeId) ?? []), issue.message]);
    }
    return map;
  }, [issues]);

  const traces = React.useMemo(() => {
    const map = new Map<string, NodeTraceState>();
    for (const trace of dryRunResult?.node_traces ?? []) {
      map.set(trace.node_id, {
        status: trace.error ? "error" : "ok",
        elapsedMs: trace.elapsed_ms,
        error: trace.error,
      });
    }
    return map;
  }, [dryRunResult]);

  const [nodes, setNodes] = React.useState<CanvasNode[]>(() =>
    deriveNodes(spec, issuesByNode, traces, []),
  );
  const [edges, setEdges] = React.useState<Edge[]>(() => deriveEdges(spec, []));

  React.useEffect(() => {
    setNodes((prev) => deriveNodes(spec, issuesByNode, traces, prev));
    setEdges((prev) => deriveEdges(spec, prev));
  }, [spec, issuesByNode, traces]);

  const specRef = React.useRef(spec);
  React.useEffect(() => {
    specRef.current = spec;
  }, [spec]);

  const onNodesChange = React.useCallback(
    (changes: Array<NodeChange<CanvasNode>>) => {
      const removals = new Set(
        changes.filter((c) => c.type === "remove").map((c) => (c as { id: string }).id),
      );
      if (removals.size > 0) {
        const cur = specRef.current;
        onSpecChange({
          nodes: cur.nodes.filter(
            (n) => !removals.has(n.id) || n.kind === "input" || n.kind === "output",
          ),
          edges: cur.edges.filter((e) => !removals.has(e.source) && !removals.has(e.target)),
        });
        setSelectedId((id) => (id && removals.has(id) ? null : id));
      }
      setNodes((prev) => applyNodeChanges(changes, prev));
      const selected = changes.find((c) => c.type === "select" && c.selected);
      if (selected) setSelectedId((selected as { id: string }).id);
      const deselected = changes.filter((c) => c.type === "select" && !c.selected);
      if (deselected.length > 0) {
        setSelectedId((id) =>
          deselected.some((c) => (c as { id: string }).id === id) ? null : id,
        );
      }
    },
    [onSpecChange],
  );

  const onEdgesChange = React.useCallback(
    (changes: Array<EdgeChange<Edge>>) => {
      const removals = new Set(
        changes.filter((c) => c.type === "remove").map((c) => (c as { id: string }).id),
      );
      if (removals.size > 0) {
        const cur = specRef.current;
        onSpecChange({
          ...cur,
          edges: cur.edges.filter((e) => !removals.has(edgeId(e))),
        });
      }
      setEdges((prev) => applyEdgeChanges(changes, prev));
    },
    [onSpecChange],
  );

  const isValidConnection: IsValidConnection<Edge> = React.useCallback(
    (conn) => {
      if (!conn.source || !conn.target || !conn.sourceHandle || !conn.targetHandle) return false;
      if (conn.source === conn.target) return false;
      const cur = specRef.current;
      const portTaken = cur.edges.some(
        (e) => e.target === conn.target && e.target_port === conn.targetHandle,
      );
      if (portTaken) return false;
      return !wouldCreateCycle(cur, conn.source, conn.target);
    },
    [],
  );

  const onConnect = React.useCallback(
    (conn: Connection) => {
      if (!conn.sourceHandle || !conn.targetHandle) return;
      const cur = specRef.current;
      onSpecChange({
        ...cur,
        edges: [
          ...cur.edges,
          {
            source: conn.source,
            source_port: conn.sourceHandle,
            target: conn.target,
            target_port: conn.targetHandle,
          },
        ],
      });
      setEdges((prev) => addEdge(conn, prev));
    },
    [onSpecChange],
  );

  const onNodeDragStop = React.useCallback(() => {
    setNodes((prev) => {
      const cur = specRef.current;
      onSpecChange({
        ...cur,
        nodes: cur.nodes.map((n) => {
          const flowNode = prev.find((fn) => fn.id === n.id);
          return flowNode
            ? { ...n, position: { x: flowNode.position.x, y: flowNode.position.y } }
            : n;
        }),
      });
      return prev;
    });
  }, [onSpecChange]);

  const addNode = React.useCallback(
    (kind: "signature" | "transform" | "mcp") => {
      const cur = specRef.current;
      const maxX = Math.max(...cur.nodes.map((n) => n.position?.x ?? 0));
      const node = newNodeSpec(kind, cur, { x: maxX / 2 + 160, y: 320 });
      onSpecChange({ ...cur, nodes: [...cur.nodes, node] });
      setSelectedId(node.id);
    },
    [onSpecChange],
  );

  const updateNode = React.useCallback(
    (next: WorkflowNodeSpec) => {
      const cur = specRef.current;
      onSpecChange({
        ...cur,
        nodes: cur.nodes.map((n) => (n.id === next.id ? next : n)),
      });
    },
    [onSpecChange],
  );

  const deleteNode = React.useCallback(
    (id: string) => {
      const cur = specRef.current;
      onSpecChange({
        nodes: cur.nodes.filter((n) => n.id !== id),
        edges: cur.edges.filter((e) => e.source !== id && e.target !== id),
      });
      setSelectedId(null);
    },
    [onSpecChange],
  );

  // Layered auto-layout: column = longest-path depth from the input anchor,
  // row = order within the column. Deterministic, no extra dependency.
  const tidyUp = React.useCallback(() => {
    const cur = specRef.current;
    const depth = new Map<string, number>(cur.nodes.map((n) => [n.id, 0]));
    for (let pass = 0; pass < cur.nodes.length; pass += 1) {
      let changed = false;
      for (const e of cur.edges) {
        const next = (depth.get(e.source) ?? 0) + 1;
        if (next > (depth.get(e.target) ?? 0)) {
          depth.set(e.target, next);
          changed = true;
        }
      }
      if (!changed) break;
    }
    const output = cur.nodes.find((n) => n.kind === "output");
    if (output) {
      const maxDepth = Math.max(...[...depth.values()]);
      depth.set(output.id, maxDepth);
    }
    const rows = new Map<number, number>();
    onSpecChange({
      ...cur,
      nodes: cur.nodes.map((n) => {
        const d = depth.get(n.id) ?? 0;
        const row = rows.get(d) ?? 0;
        rows.set(d, row + 1);
        return { ...n, position: { x: d * 300, y: row * 160 + (d % 2) * 40 } };
      }),
    });
    window.setTimeout(() => fitView({ padding: 0.2, duration: 200 }), 50);
  }, [onSpecChange, fitView]);

  const toggleFullscreen = React.useCallback(() => {
    setFullscreen((f) => !f);
    window.setTimeout(() => fitView({ padding: 0.2 }), 60);
  }, [fitView]);

  const selectedNode = spec.nodes.find((n) => n.id === selectedId) ?? null;
  const inputAnchor = spec.nodes.find((n) => n.kind === "input");
  const inputFieldNames = inputAnchor ? nodePorts(inputAnchor).outputs.map((p) => p.name) : [];

  return (
    <div
      className={cn(
        fullscreen
          ? "fixed inset-0 z-50 flex flex-col bg-background p-3"
          : "relative flex flex-col",
        className,
      )}
      data-tutorial="workflow-canvas"
    >
      <div className="flex flex-wrap items-center gap-1.5 border-b border-border/40 bg-[#FAF8F5] px-3 py-2">
        <ToolbarButton icon={Sparkles} label={msg("workflow.toolbar.add_signature")} onClick={() => addNode("signature")} />
        <ToolbarButton icon={Code2} label={msg("workflow.toolbar.add_transform")} onClick={() => addNode("transform")} />
        <ToolbarButton icon={Wrench} label={msg("workflow.toolbar.add_tool")} onClick={() => addNode("mcp")} />
        <div className="mx-1 h-4 w-px bg-border" />
        <ToolbarButton icon={LayoutGrid} label={msg("workflow.toolbar.tidy")} onClick={tidyUp} />
        <span className="ms-auto" />
        {issues.length > 0 && (
          <span className="inline-flex items-center gap-1 text-[0.6875rem] font-medium text-[#A3512B]">
            <AlertTriangle className="size-3" />
            {formatMsg("workflow.toolbar.issues", { p1: issues.length })}
          </span>
        )}
        {dryRun && (
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1.5 text-xs"
            disabled={issues.length > 0 || !!dryRun.disabledReason}
            title={dryRun.disabledReason ?? undefined}
            onClick={() => setDryRunOpen(true)}
          >
            <Play className="size-3" />
            {msg("workflow.toolbar.dry_run")}
          </Button>
        )}
        <ToolbarButton
          icon={fullscreen ? Minimize2 : Maximize2}
          label={msg(fullscreen ? "workflow.toolbar.exit_fullscreen" : "workflow.toolbar.fullscreen")}
          onClick={toggleFullscreen}
          iconOnly
        />
      </div>

      <div
        className={cn(
          "grid min-h-0 flex-1",
          selectedNode ? "grid-cols-[minmax(0,1fr)_320px]" : "grid-cols-1",
        )}
      >
        <div dir="ltr" className={cn("relative", fullscreen ? "min-h-0" : "h-[480px]")}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDragStop={onNodeDragStop}
            isValidConnection={isValidConnection}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            proOptions={{ hideAttribution: true }}
            deleteKeyCode={["Backspace", "Delete"]}
            className="bg-[#FDFCFA]"
          >
            <Background gap={20} size={1} color="#E5DDD1" />
            <Controls showInteractive={false} position="bottom-left" />
          </ReactFlow>
        </div>
        {selectedNode && (
          <div className="min-h-0 overflow-hidden border-s border-border/40">
            <NodeInspector
              spec={selectedNode}
              issues={issuesByNode.get(selectedNode.id) ?? []}
              onChange={updateNode}
              onDelete={
                selectedNode.kind !== "input" && selectedNode.kind !== "output"
                  ? () => deleteNode(selectedNode.id)
                  : undefined
              }
            />
          </div>
        )}
      </div>

      {dryRunResult && (
        <DryRunResultBar result={dryRunResult} onDismiss={() => setDryRunResult(null)} />
      )}

      {dryRun && (
        <DryRunDialog
          open={dryRunOpen}
          onOpenChange={setDryRunOpen}
          inputFields={inputFieldNames}
          sampleInputs={dryRun.sampleInputs}
          run={dryRun.run}
          onResult={setDryRunResult}
        />
      )}
    </div>
  );
}

function ToolbarButton({
  icon: Icon,
  label,
  onClick,
  iconOnly = false,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick: () => void;
  iconOnly?: boolean;
}) {
  return (
    <Button
      size="sm"
      variant="ghost"
      className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
      onClick={onClick}
      title={iconOnly ? label : undefined}
      aria-label={label}
    >
      <Icon className="size-3.5" />
      {!iconOnly && label}
    </Button>
  );
}

function DryRunResultBar({
  result,
  onDismiss,
}: {
  result: WorkflowDryRunResponse;
  onDismiss: () => void;
}) {
  const failed = !!result.error;
  return (
    <div
      className={cn(
        "flex items-start gap-3 border-t px-4 py-2.5 text-xs",
        failed ? "border-destructive/30 bg-[#FBF0EC]" : "border-border/40 bg-[#F5F8F1]",
      )}
    >
      <span className={cn("mt-0.5 font-semibold", failed ? "text-destructive" : "text-[#5A7247]")}>
        {failed ? msg("workflow.dryrun.failed") : msg("workflow.dryrun.succeeded")}
      </span>
      <div className="min-w-0 flex-1 space-y-0.5">
        {failed ? (
          <span className="break-words text-muted-foreground" dir="ltr">
            {result.error}
          </span>
        ) : (
          Object.entries(result.outputs ?? {}).map(([key, value]) => (
            <div key={key} className="truncate text-muted-foreground" dir="ltr">
              <span className="font-mono font-medium text-foreground">{key}</span>
              {": "}
              {String(value)}
            </div>
          ))
        )}
      </div>
      <button
        type="button"
        onClick={onDismiss}
        className="shrink-0 text-muted-foreground hover:text-foreground"
        aria-label={msg("workflow.dryrun.dismiss")}
      >
        <X className="size-3.5" />
      </button>
    </div>
  );
}

export function WorkflowCanvas({ specRevision, ...props }: WorkflowCanvasProps) {
  return (
    <ReactFlowProvider key={specRevision}>
      <CanvasInner {...props} />
    </ReactFlowProvider>
  );
}
