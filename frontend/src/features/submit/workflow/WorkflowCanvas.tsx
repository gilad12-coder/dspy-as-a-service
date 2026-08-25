"use client";

/**
 * The workflow canvas: an n8n-style node editor over the wizard's
 * `WorkflowSpec`.
 *
 * The spec is the single source of truth (owned by the wizard hook); React
 * Flow state is re-derived from it after every structural mutation. Pure
 * drag movement stays inside React Flow until drag-stop, when positions are
 * committed back onto the spec. External spec replacements (dataset init,
 * clone hydration, agent ops) bump `specRevision`, which remounts the flow
 * so stale internal state can't survive.
 *
 * Interaction model: scroll (or pinch) zooms, left-drag pans, Shift+drag
 * box-selects. A single click only selects a node; double-click opens the
 * inspector, so dragging never accidentally opens configuration. Ctrl/⌘+Z
 * undoes graph edits (Shift+Z / Y redoes), Ctrl/⌘+C/V copy-paste selected
 * nodes. Nodes are added from a context menu (pane right-click, the toolbar
 * "Add node" button, or dropping a half-made connection on empty canvas —
 * the latter auto-wires the new node). Fullscreen portals the whole editor
 * onto `document.body` so no transformed/filtered ancestor can trap it.
 *
 * The canvas itself is always LTR — data flows left→right — while inspector
 * chrome follows the page direction.
 */

import * as React from "react";
import { createPortal } from "react-dom";
import {
  Background,
  ConnectionLineType,
  MarkerType,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  useReactFlow,
  useStore,
  type Connection,
  type Edge,
  type EdgeChange,
  type FinalConnectionState,
  type IsValidConnection,
  type NodeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { AnimatePresence, motion } from "framer-motion";
import { toast } from "react-toastify";
import {
  ArrowCounterClockwise,
  ArrowsIn,
  ArrowsOut,
  CaretDown,
  Code,
  Copy,
  GridFour,
  MagnifyingGlassMinus,
  MagnifyingGlassPlus,
  Play,
  Plus,
  Robot,
  Sparkle,
  Trash,
  Warning,
  Wrench,
  X,
} from "@/shared/ui/icons";

import { Button } from "@/shared/ui/primitives/button";
import { cn } from "@/shared/lib/utils";
import type { WorkflowDryRunStreamHandlers } from "@/shared/lib/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import type { WorkflowDryRunResponse, WorkflowNodeSpec, WorkflowSpec } from "@/shared/types/api";

import {
  autoLayoutSpec,
  makeNodeId,
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
  /** True when no model is configured yet (the model step comes later). */
  needsModel?: boolean;
  /** Opens the wizard's model-config modal so the user can pick in place. */
  pickModel?: () => void;
  /** Name of the currently selected model, shown (and changeable) in the
      dry-run dialog. */
  modelName?: string | null;
  /** Prefill values for the input anchor's fields (first dataset row). */
  sampleInputs: Record<string, string>;
  run: (inputs: Record<string, unknown>, handlers: WorkflowDryRunStreamHandlers) => Promise<void>;
}

interface WorkflowCanvasProps {
  spec: WorkflowSpec;
  specRevision: number;
  onSpecChange: (spec: WorkflowSpec) => void;
  dryRun?: WorkflowDryRunBinding;
  // Node the code agent just changed — pulsed briefly on the canvas.
  pulseNodeId?: string | null;
  // Agent chat surface for fullscreen mode, where the wizard's own panel is
  // hidden behind the portal. Rendered as a collapsible start-side overlay.
  agentPanel?: React.ReactNode;
  className?: string;
}

const SNAP_GRID: [number, number] = [16, 16];
const FIT_VIEW = { padding: 0.2, maxZoom: 1 };
const EDGE_COLOR = "#8A7563";

const ADD_KINDS = [
  { kind: "signature", icon: Sparkle, labelKey: "workflow.toolbar.add_signature" },
  { kind: "transform", icon: Code, labelKey: "workflow.toolbar.add_transform" },
  { kind: "mcp", icon: Wrench, labelKey: "workflow.toolbar.add_tool" },
] as const;

type MenuTarget = { type: "pane" } | { type: "node"; id: string } | { type: "edge"; id: string };

interface MenuState {
  // Position relative to the canvas root: `x` is a logical inline-start
  // offset (so the menu opens toward the inline-end in both LTR and RTL),
  // `y` a top offset. Both in px.
  x: number;
  y: number;
  // Where an added node lands, in flow coordinates.
  flow: { x: number; y: number };
  target: MenuTarget;
  // Set when the menu opened from a connection dropped on empty canvas —
  // the added node gets auto-wired to this handle.
  pending: { nodeId: string; handleId: string; dir: "out" | "in" } | null;
}

const edgeId = (e: { source: string; source_port: string; target: string; target_port: string }) =>
  `${e.source}.${e.source_port}->${e.target}.${e.target_port}`;

// Position the canvas menu by a logical inline-start offset within the root,
// so it opens toward the inline-end — rightward in LTR, leftward in RTL —
// instead of always growing to the physical right, which ran it off-screen in
// RTL. Clamp to a conservative width estimate so the far edge stays visible.
const menuInlineStart = (physicalX: number, rootRect: DOMRect, isRtl: boolean): number => {
  const start = isRtl ? rootRect.right - physicalX : physicalX - rootRect.left;
  return Math.max(4, Math.min(start, rootRect.width - 200));
};

function deriveNodes(
  spec: WorkflowSpec,
  issuesByNode: Map<string, string[]>,
  traces: Map<string, NodeTraceState>,
  prev: CanvasNode[],
  pulseNodeId: string | null,
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
      pulse: node.id === pulseNodeId,
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
    style: { stroke: EDGE_COLOR, strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLOR, width: 16, height: 16 },
    interactionWidth: 40,
  }));
}

function CanvasInner({
  spec,
  onSpecChange,
  dryRun,
  pulseNodeId = null,
  agentPanel,
  className,
}: Omit<WorkflowCanvasProps, "specRevision">) {
  const { fitView, screenToFlowPosition } = useReactFlow();
  const [fullscreen, setFullscreen] = React.useState(false);
  // Fullscreen-only agent chat overlay; remembered across fullscreen toggles.
  const [agentOpen, setAgentOpen] = React.useState(false);
  // Which node's configuration panel is open. Deliberately decoupled from
  // canvas selection: single click selects/drags, double-click inspects.
  const [inspectorId, setInspectorId] = React.useState<string | null>(null);
  const [dryRunOpen, setDryRunOpen] = React.useState(false);
  const [dryRunResult, setDryRunResult] = React.useState<WorkflowDryRunResponse | null>(null);
  const [menu, setMenu] = React.useState<MenuState | null>(null);
  // Minimap fades in while the viewport is moving and back out shortly after.
  const [navActive, setNavActive] = React.useState(false);
  const navTimerRef = React.useRef<number | null>(null);
  const rootRef = React.useRef<HTMLDivElement | null>(null);
  const flowWrapRef = React.useRef<HTMLDivElement | null>(null);
  const menuRef = React.useRef<HTMLDivElement | null>(null);
  const addNodeBtnRef = React.useRef<HTMLButtonElement | null>(null);
  const menuOpenRef = React.useRef(false);
  React.useEffect(() => {
    menuOpenRef.current = menu !== null;
  }, [menu]);

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
    deriveNodes(spec, issuesByNode, traces, [], pulseNodeId),
  );
  const [edges, setEdges] = React.useState<Edge[]>(() => deriveEdges(spec, []));

  React.useEffect(() => {
    setNodes((prev) => deriveNodes(spec, issuesByNode, traces, prev, pulseNodeId));
    setEdges((prev) => deriveEdges(spec, prev));
  }, [spec, issuesByNode, traces, pulseNodeId]);

  const specRef = React.useRef(spec);
  React.useEffect(() => {
    specRef.current = spec;
  }, [spec]);
  const nodesRef = React.useRef<CanvasNode[]>([]);
  React.useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  // Undo/redo over spec snapshots. Every local edit goes through commit();
  // external replacements (agent ops, clone hydration) bump specRevision and
  // remount the canvas, which intentionally resets history to that baseline.
  const pastRef = React.useRef<WorkflowSpec[]>([]);
  const futureRef = React.useRef<WorkflowSpec[]>([]);
  // Coalesces bursts with the same key (inspector keystrokes) into one entry.
  const lastCommitRef = React.useRef<{ key: string; at: number } | null>(null);
  const commit = React.useCallback(
    (next: WorkflowSpec, coalesceKey?: string) => {
      const now = Date.now();
      const last = lastCommitRef.current;
      const coalesce = !!coalesceKey && !!last && last.key === coalesceKey && now - last.at < 1200;
      if (!coalesce) {
        pastRef.current.push(specRef.current);
        if (pastRef.current.length > 100) pastRef.current.shift();
        futureRef.current = [];
      }
      lastCommitRef.current = coalesceKey ? { key: coalesceKey, at: now } : null;
      onSpecChange(next);
    },
    [onSpecChange],
  );
  const undo = React.useCallback(() => {
    const prev = pastRef.current.pop();
    if (!prev) return;
    futureRef.current.push(specRef.current);
    lastCommitRef.current = null;
    onSpecChange(prev);
  }, [onSpecChange]);
  const redo = React.useCallback(() => {
    const next = futureRef.current.pop();
    if (!next) return;
    pastRef.current.push(specRef.current);
    lastCommitRef.current = null;
    onSpecChange(next);
  }, [onSpecChange]);

  const onNodesChange = React.useCallback(
    (changes: Array<NodeChange<CanvasNode>>) => {
      const removals = new Set(
        changes.filter((c) => c.type === "remove").map((c) => (c as { id: string }).id),
      );
      if (removals.size > 0) {
        const cur = specRef.current;
        commit({
          nodes: cur.nodes.filter(
            (n) => !removals.has(n.id) || n.kind === "input" || n.kind === "output",
          ),
          edges: cur.edges.filter((e) => !removals.has(e.source) && !removals.has(e.target)),
        });
        setInspectorId((id) => (id && removals.has(id) ? null : id));
      }
      setNodes((prev) => applyNodeChanges(changes, prev));
    },
    [commit],
  );

  const onEdgesChange = React.useCallback(
    (changes: Array<EdgeChange<Edge>>) => {
      const removals = new Set(
        changes.filter((c) => c.type === "remove").map((c) => (c as { id: string }).id),
      );
      if (removals.size > 0) {
        const cur = specRef.current;
        commit({
          ...cur,
          edges: cur.edges.filter((e) => !removals.has(edgeId(e))),
        });
      }
      setEdges((prev) => applyEdgeChanges(changes, prev));
    },
    [commit],
  );

  const isValidConnection: IsValidConnection<Edge> = React.useCallback((conn) => {
    if (!conn.source || !conn.target || !conn.sourceHandle || !conn.targetHandle) return false;
    if (conn.source === conn.target) return false;
    const cur = specRef.current;
    const portTaken = cur.edges.some(
      (e) => e.target === conn.target && e.target_port === conn.targetHandle,
    );
    if (portTaken) return false;
    return !wouldCreateCycle(cur, conn.source, conn.target);
  }, []);

  const onConnect = React.useCallback(
    (conn: Connection) => {
      if (!conn.sourceHandle || !conn.targetHandle) return;
      const cur = specRef.current;
      commit({
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
    [commit],
  );

  // A connection dropped on empty canvas opens the add-node menu at the drop
  // point; the picked node is wired straight to the dangling handle (n8n's
  // drag-to-spawn gesture).
  const onConnectEnd = React.useCallback(
    (event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
      if (state.isValid || !state.fromNode || !state.fromHandle) return;
      const targetEl = event.target as Element | null;
      if (!targetEl?.closest(".react-flow__pane")) return;
      const point = "changedTouches" in event ? event.changedTouches[0] : event;
      const root = rootRef.current;
      if (!root || !point) return;
      const rect = root.getBoundingClientRect();
      const isRtl = document.documentElement.dir === "rtl";
      setMenu({
        x: menuInlineStart(point.clientX, rect, isRtl),
        y: Math.min(point.clientY - rect.top, rect.height - 170),
        flow: screenToFlowPosition({ x: point.clientX, y: point.clientY }),
        target: { type: "pane" },
        pending: {
          nodeId: state.fromNode.id,
          handleId: state.fromHandle.id ?? "",
          dir: state.fromHandle.type === "source" ? "out" : "in",
        },
      });
    },
    [screenToFlowPosition],
  );

  // Commit dragged positions from the event args — never from inside a
  // setNodes updater, which React may run during render (setState-in-render).
  const onNodeDragStop = React.useCallback(
    (_event: MouseEvent | TouchEvent, _node: CanvasNode, dragged: CanvasNode[]) => {
      const moved = new Map(dragged.map((n) => [n.id, n.position]));
      if (moved.size === 0) return;
      const cur = specRef.current;
      commit({
        ...cur,
        nodes: cur.nodes.map((n) => {
          const pos = moved.get(n.id);
          return pos ? { ...n, position: { x: pos.x, y: pos.y } } : n;
        }),
      });
    },
    [commit],
  );

  const addNodeAt = React.useCallback(
    (
      kind: "signature" | "transform" | "mcp",
      position: { x: number; y: number },
      pending: MenuState["pending"],
    ) => {
      const cur = specRef.current;
      const node = newNodeSpec(kind, cur, {
        x: Math.round(position.x / SNAP_GRID[0]) * SNAP_GRID[0],
        y: Math.round(position.y / SNAP_GRID[1]) * SNAP_GRID[1],
      });
      let nextEdges = cur.edges;
      if (pending) {
        const ports = nodePorts(node);
        if (pending.dir === "out") {
          const target = ports.inputs[0];
          if (target) {
            nextEdges = [
              ...nextEdges,
              {
                source: pending.nodeId,
                source_port: pending.handleId,
                target: node.id,
                target_port: target.name,
              },
            ];
          }
        } else {
          // Dragged backwards from an input handle — only wire it if that
          // port isn't already fed (single-producer rule).
          const source = ports.outputs[0];
          const fed = cur.edges.some(
            (e) => e.target === pending.nodeId && e.target_port === pending.handleId,
          );
          if (source && !fed) {
            nextEdges = [
              ...nextEdges,
              {
                source: node.id,
                source_port: source.name,
                target: pending.nodeId,
                target_port: pending.handleId,
              },
            ];
          }
        }
      }
      commit({ nodes: [...cur.nodes, node], edges: nextEdges });
      setInspectorId(node.id);
    },
    [commit],
  );

  const duplicateNode = React.useCallback(
    (id: string) => {
      const cur = specRef.current;
      const orig = cur.nodes.find((n) => n.id === id);
      if (!orig || orig.kind === "input" || orig.kind === "output") return;
      const prefix =
        orig.kind === "signature" ? "step" : orig.kind === "mcp" ? "tool" : "transform";
      const copy: WorkflowNodeSpec = JSON.parse(JSON.stringify(orig));
      copy.id = makeNodeId(prefix, cur);
      copy.position = {
        x: (orig.position?.x ?? 0) + 48,
        y: (orig.position?.y ?? 0) + 48,
      };
      commit({ ...cur, nodes: [...cur.nodes, copy] });
    },
    [commit],
  );

  const updateNode = React.useCallback(
    (next: WorkflowNodeSpec) => {
      const cur = specRef.current;
      commit(
        {
          ...cur,
          nodes: cur.nodes.map((n) => (n.id === next.id ? next : n)),
        },
        // Inspector edits arrive per keystroke; fold a burst into one undo step.
        `update:${next.id}`,
      );
    },
    [commit],
  );

  const deleteNode = React.useCallback(
    (id: string) => {
      const cur = specRef.current;
      commit({
        nodes: cur.nodes.filter((n) => n.id !== id),
        edges: cur.edges.filter((e) => e.source !== id && e.target !== id),
      });
      setInspectorId((open) => (open === id ? null : open));
    },
    [commit],
  );

  const deleteEdge = React.useCallback(
    (id: string) => {
      const cur = specRef.current;
      commit({ ...cur, edges: cur.edges.filter((e) => edgeId(e) !== id) });
    },
    [commit],
  );

  const tidyUp = React.useCallback(() => {
    commit(autoLayoutSpec(specRef.current));
    window.setTimeout(() => fitView({ ...FIT_VIEW, duration: 300 }), 50);
  }, [commit, fitView]);

  // Clipboard for node copy/paste. Internal edges (both endpoints copied)
  // come along; ids are re-minted on paste so copies never collide.
  const clipboardRef = React.useRef<{
    nodes: WorkflowNodeSpec[];
    edges: WorkflowSpec["edges"];
  } | null>(null);

  const copySelection = React.useCallback(() => {
    const picked = nodesRef.current
      .filter((n) => n.selected)
      .map((n) => n.data.spec)
      .filter((s) => s.kind !== "input" && s.kind !== "output");
    if (picked.length === 0) return false;
    const ids = new Set(picked.map((s) => s.id));
    const cur = specRef.current;
    clipboardRef.current = JSON.parse(
      JSON.stringify({
        nodes: picked,
        edges: cur.edges.filter((e) => ids.has(e.source) && ids.has(e.target)),
      }),
    ) as { nodes: WorkflowNodeSpec[]; edges: WorkflowSpec["edges"] };
    return true;
  }, []);

  const pasteClipboard = React.useCallback(() => {
    const clip = clipboardRef.current;
    if (!clip || clip.nodes.length === 0) return;
    const cur = specRef.current;
    let scratch = cur;
    const idMap = new Map<string, string>();
    const copies = clip.nodes.map((n) => {
      const prefix = n.kind === "signature" ? "step" : n.kind === "mcp" ? "tool" : "transform";
      const copy = JSON.parse(JSON.stringify(n)) as WorkflowNodeSpec;
      copy.id = makeNodeId(prefix, scratch);
      idMap.set(n.id, copy.id);
      copy.position = { x: (n.position?.x ?? 0) + 48, y: (n.position?.y ?? 0) + 48 };
      scratch = { ...scratch, nodes: [...scratch.nodes, copy] };
      return copy;
    });
    const copiedEdges = clip.edges.map((e) => ({
      ...e,
      source: idMap.get(e.source) ?? e.source,
      target: idMap.get(e.target) ?? e.target,
    }));
    commit({ nodes: [...cur.nodes, ...copies], edges: [...cur.edges, ...copiedEdges] });
  }, [commit]);

  // Editor shortcuts: Ctrl/⌘+Z undo, Shift+Z / Y redo, C/V copy-paste nodes.
  // Skips events aimed at text inputs and CodeMirror so their own shortcuts
  // (including the code editors' undo) still win.
  React.useEffect(() => {
    const inTextTarget = (t: EventTarget | null) =>
      t instanceof HTMLElement &&
      t.closest("input, textarea, select, [contenteditable=true], .cm-editor") !== null;
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || inTextTarget(e.target)) return;
      const key = e.key.toLowerCase();
      if (key === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if (key === "y") {
        e.preventDefault();
        redo();
      } else if (key === "c") {
        // Leave real text copying alone.
        if (window.getSelection()?.isCollapsed === false) return;
        if (copySelection()) e.preventDefault();
      } else if (key === "v") {
        if (clipboardRef.current) {
          e.preventDefault();
          pasteClipboard();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, copySelection, pasteClipboard]);

  // If an undo or agent change removed the inspected node, close the panel.
  React.useEffect(() => {
    setInspectorId((id) => (id && !spec.nodes.some((n) => n.id === id) ? null : id));
  }, [spec]);

  const openMenuAt = React.useCallback(
    (clientX: number, clientY: number, target: MenuTarget) => {
      const root = rootRef.current;
      if (!root) return;
      const rect = root.getBoundingClientRect();
      const isRtl = document.documentElement.dir === "rtl";
      setMenu({
        x: menuInlineStart(clientX, rect, isRtl),
        y: Math.min(clientY - rect.top, rect.height - 170),
        flow: screenToFlowPosition({ x: clientX, y: clientY }),
        target,
        pending: null,
      });
    },
    [screenToFlowPosition],
  );

  const onPaneContextMenu = React.useCallback(
    (event: MouseEvent | React.MouseEvent) => {
      event.preventDefault();
      openMenuAt(event.clientX, event.clientY, { type: "pane" });
    },
    [openMenuAt],
  );

  const onNodeContextMenu = React.useCallback(
    (event: React.MouseEvent, node: CanvasNode) => {
      event.preventDefault();
      const kind = node.data.spec.kind;
      if (kind === "input" || kind === "output") return;
      openMenuAt(event.clientX, event.clientY, { type: "node", id: node.id });
    },
    [openMenuAt],
  );

  const onEdgeContextMenu = React.useCallback(
    (event: React.MouseEvent, edge: Edge) => {
      event.preventDefault();
      openMenuAt(event.clientX, event.clientY, { type: "edge", id: edge.id });
    },
    [openMenuAt],
  );

  const openAddMenuFromToolbar = React.useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      const root = rootRef.current;
      const flowEl = flowWrapRef.current;
      if (!root || !flowEl) return;
      const rootRect = root.getBoundingClientRect();
      const btnRect = event.currentTarget.getBoundingClientRect();
      const flowRect = flowEl.getBoundingClientRect();
      const isRtl = document.documentElement.dir === "rtl";
      // Anchor to the button's inline-start edge so the menu drops down aligned
      // under it and opens toward the inline-end.
      const anchorX = menuInlineStart(isRtl ? btnRect.right : btnRect.left, rootRect, isRtl);
      setMenu((prev) =>
        prev
          ? null
          : {
              x: anchorX,
              y: btnRect.bottom - rootRect.top + 4,
              flow: screenToFlowPosition({
                x: flowRect.left + flowRect.width / 2,
                y: flowRect.top + flowRect.height / 2,
              }),
              target: { type: "pane" },
              pending: null,
            },
      );
    },
    [screenToFlowPosition],
  );

  const onNodeDoubleClick = React.useCallback((_event: React.MouseEvent, node: CanvasNode) => {
    setInspectorId(node.id);
  }, []);

  const onNodeClick = React.useCallback((_event: React.MouseEvent, node: CanvasNode) => {
    if (!window.matchMedia("(any-pointer: coarse)").matches) return;
    setInspectorId(node.id);
  }, []);

  const onEdgeClick = React.useCallback(
    (event: React.MouseEvent, edge: Edge) => {
      if (!window.matchMedia("(any-pointer: coarse)").matches) return;
      event.preventDefault();
      openMenuAt(event.clientX, event.clientY, { type: "edge", id: edge.id });
    },
    [openMenuAt],
  );

  const onPaneClick = React.useCallback(() => {
    setMenu(null);
    setInspectorId(null);
  }, []);

  // Dismiss the menu on any pointer-down outside it (capture phase so canvas
  // interactions can't swallow the event first). The toolbar's add-node
  // trigger is excluded: it toggles the menu itself, so letting this handler
  // close it first would close-then-reopen and the menu could never collapse.
  React.useEffect(() => {
    if (!menu) return;
    const onPointerDown = (e: PointerEvent) => {
      const el = menuRef.current;
      const btn = addNodeBtnRef.current;
      if (!(e.target instanceof globalThis.Node)) return;
      if (btn?.contains(e.target)) return;
      if (el && !el.contains(e.target)) setMenu(null);
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => document.removeEventListener("pointerdown", onPointerDown, true);
  }, [menu]);

  // ESC closes the menu first, then the inspector, then the agent overlay,
  // then exits fullscreen — like a modal stack.
  const inspectorOpenRef = React.useRef(false);
  React.useEffect(() => {
    inspectorOpenRef.current = inspectorId !== null;
  }, [inspectorId]);
  const agentOpenRef = React.useRef(false);
  React.useEffect(() => {
    agentOpenRef.current = agentOpen;
  }, [agentOpen]);
  React.useEffect(() => {
    if (!menu && !fullscreen && !inspectorId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // The agent composer handles its own ESC (e.g. cancel an edit); don't
      // steal it while the user is typing there.
      if (e.target instanceof Element && e.target.closest("input, textarea")) return;
      e.preventDefault();
      if (menuOpenRef.current) setMenu(null);
      else if (inspectorOpenRef.current) setInspectorId(null);
      else if (agentOpenRef.current) setAgentOpen(false);
      else setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menu, fullscreen, inspectorId]);

  // Lock page scroll while the fullscreen overlay is open (portal pattern
  // shared with the trajectory tree).
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

  const handleMoveStart = React.useCallback(() => {
    if (navTimerRef.current) window.clearTimeout(navTimerRef.current);
    setNavActive(true);
    setMenu(null);
  }, []);
  const handleMoveEnd = React.useCallback(() => {
    if (navTimerRef.current) window.clearTimeout(navTimerRef.current);
    navTimerRef.current = window.setTimeout(() => setNavActive(false), 1000);
  }, []);

  const inspectorNode = spec.nodes.find((n) => n.id === inspectorId) ?? null;
  // The inspector slides in from the inline end; page direction decides which
  // physical side that is. Client-only component, so document is available.
  const slideFrom = document.documentElement.dir === "rtl" ? -24 : 24;
  const inputAnchor = spec.nodes.find((n) => n.kind === "input");
  const inputFieldNames = inputAnchor ? nodePorts(inputAnchor).outputs.map((p) => p.name) : [];
  const outputAnchor = spec.nodes.find((n) => n.kind === "output");
  const outputFieldNames = outputAnchor ? nodePorts(outputAnchor).inputs.map((p) => p.name) : [];

  const menuNode = menu?.target.type === "node" ? menu.target.id : null;
  const menuEdge = menu?.target.type === "edge" ? menu.target.id : null;

  const body = (
    <div
      ref={rootRef}
      className={cn(
        fullscreen
          ? "fixed inset-0 z-50 flex h-screen w-screen flex-col bg-background"
          : "relative flex flex-col",
        className,
      )}
      data-tutorial="workflow-canvas"
    >
      <div className="flex flex-wrap items-center gap-1.5 border-b border-border/40 bg-[#FAF8F5] px-3 py-2">
        {/* Fullscreen swaps the start-side editing buttons for the agent
            toggle: node adding + tidy stay reachable via the context menu,
            and the agent panel toggles like the generalist panel does —
            without leaving fullscreen, over the same shared conversation. */}
        {fullscreen ? (
          agentPanel && (
            <ToolbarButton
              icon={Robot}
              label={msg("workflow.toolbar.agent")}
              onClick={() => setAgentOpen((v) => !v)}
              active={agentOpen}
            />
          )
        ) : (
          <>
            <Button
              ref={addNodeBtnRef}
              size="sm"
              variant="outline"
              className="h-[44px] gap-1.5 px-2.5 text-xs lg:h-7"
              onClick={openAddMenuFromToolbar}
            >
              <Plus className="size-3.5" />
              {msg("workflow.toolbar.add_node")}
              <CaretDown className="size-3 text-muted-foreground" />
            </Button>
            <ToolbarButton icon={GridFour} label={msg("workflow.toolbar.tidy")} onClick={tidyUp} />
          </>
        )}
        <span className="ms-auto" />
        {issues.length > 0 && (
          <span className="inline-flex items-center gap-1 text-[0.6875rem] font-medium text-[#A3512B]">
            <Warning className="size-3" />
            {formatMsg("workflow.toolbar.issues", { p1: issues.length })}
          </span>
        )}
        {dryRun && (
          <Button
            size="sm"
            variant="outline"
            className="h-[44px] gap-1.5 text-xs lg:h-7"
            title={dryRun.disabledReason ?? undefined}
            onClick={() => {
              // Always clickable: a blocked run explains itself instead of
              // silently ignoring the click behind a disabled button.
              const firstIssue = issues[0];
              if (firstIssue) {
                toast.error(firstIssue.message);
                return;
              }
              if (dryRun.disabledReason) {
                toast.error(dryRun.disabledReason);
                return;
              }
              // No model yet (that step comes later) — open the picker right
              // here instead of bouncing the user to a future step.
              if (dryRun.needsModel && dryRun.pickModel) {
                dryRun.pickModel();
                return;
              }
              setDryRunOpen(true);
            }}
          >
            <Play className="size-3" />
            {msg("workflow.toolbar.dry_run")}
          </Button>
        )}
        <ToolbarButton
          icon={fullscreen ? ArrowsIn : ArrowsOut}
          label={msg(
            fullscreen ? "workflow.toolbar.exit_fullscreen" : "workflow.toolbar.fullscreen",
          )}
          onClick={() => setFullscreen((f) => !f)}
          iconOnly
        />
      </div>

      {/* The inspector overlays the canvas (slides in from the inline end)
          instead of claiming a grid column, so opening it never resizes or
          re-fits the graph. In fullscreen the agent chat mirrors it on the
          inline start. */}
      <div className="relative flex min-h-0 flex-1">
        <AnimatePresence>
          {fullscreen && agentPanel && agentOpen && (
            <motion.div
              key="agent-panel"
              initial={{ x: -slideFrom, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -slideFrom, opacity: 0 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
              className="absolute inset-y-0 start-0 z-20 w-[min(400px,90vw)] overflow-hidden border-e border-border/40 bg-card shadow-xl"
            >
              {agentPanel}
            </motion.div>
          )}
        </AnimatePresence>
        <div
          dir="ltr"
          ref={flowWrapRef}
          className={cn("min-w-0 flex-1", fullscreen ? "min-h-0" : "h-[480px]")}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onConnectEnd={onConnectEnd}
            onNodeDragStop={onNodeDragStop}
            onNodeDoubleClick={onNodeDoubleClick}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            isValidConnection={isValidConnection}
            onPaneContextMenu={onPaneContextMenu}
            onNodeContextMenu={onNodeContextMenu}
            onEdgeContextMenu={onEdgeContextMenu}
            onPaneClick={onPaneClick}
            onMoveStart={handleMoveStart}
            onMoveEnd={handleMoveEnd}
            minZoom={0.1}
            maxZoom={3}
            zoomOnDoubleClick={false}
            snapToGrid
            snapGrid={SNAP_GRID}
            connectionRadius={60}
            connectionLineType={ConnectionLineType.Bezier}
            connectionLineStyle={{ stroke: EDGE_COLOR, strokeWidth: 2 }}
            fitView
            fitViewOptions={FIT_VIEW}
            proOptions={{ hideAttribution: true }}
            deleteKeyCode={["Backspace", "Delete"]}
            className="bg-[#FDFCFA]"
          >
            <Background gap={16} size={1.25} color="#E3D9CB" />
            <ZoomControls />
            <Panel position="bottom-center" className="pointer-events-none !m-3">
              <span
                dir="auto"
                className="hidden whitespace-nowrap rounded-full border border-border/50 bg-background/80 px-3 py-1 text-[0.6875rem] text-muted-foreground/80 shadow-xs backdrop-blur sm:inline-flex"
              >
                {msg("workflow.canvas.gestures")}
              </span>
            </Panel>
            <MiniMap
              pannable
              zoomable
              position="bottom-right"
              nodeColor={(n) =>
                n.type === "input_anchor" || n.type === "output_anchor" ? "#C8A882" : "#DDD2C2"
              }
              nodeStrokeColor="#B9A78F"
              nodeBorderRadius={10}
              maskColor="rgba(61, 46, 34, 0.06)"
              bgColor="#FAF8F5"
              className={cn(
                "!m-3 hidden overflow-hidden rounded-lg border border-border/60 shadow-sm transition-opacity duration-300 sm:block",
                navActive ? "opacity-100" : "pointer-events-none opacity-0",
              )}
            />
          </ReactFlow>
        </div>
        <AnimatePresence>
          {inspectorNode && (
            <motion.div
              key="inspector"
              initial={{ x: slideFrom, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: slideFrom, opacity: 0 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
              className={cn(
                "absolute inset-y-0 end-0 z-20 overflow-hidden border-s border-border/40 bg-card shadow-xl",
                fullscreen ? "w-full sm:w-[min(360px,88vw)]" : "w-full sm:w-[min(320px,88vw)]",
              )}
            >
              <NodeInspector
                onClose={() => setInspectorId(null)}
                spec={inspectorNode}
                issues={issuesByNode.get(inspectorNode.id) ?? []}
                onChange={updateNode}
                onDelete={
                  inspectorNode.kind !== "input" && inspectorNode.kind !== "output"
                    ? () => deleteNode(inspectorNode.id)
                    : undefined
                }
                onDuplicate={
                  inspectorNode.kind !== "input" && inspectorNode.kind !== "output"
                    ? () => duplicateNode(inspectorNode.id)
                    : undefined
                }
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {dryRunResult && (
        <DryRunResultBar result={dryRunResult} onDismiss={() => setDryRunResult(null)} />
      )}

      {dryRun && (
        <DryRunDialog
          open={dryRunOpen}
          onOpenChange={setDryRunOpen}
          inputFields={inputFieldNames}
          outputFields={outputFieldNames}
          sampleInputs={dryRun.sampleInputs}
          modelName={dryRun.modelName ?? null}
          onPickModel={dryRun.pickModel}
          run={dryRun.run}
          onResult={setDryRunResult}
        />
      )}

      <AnimatePresence>
        {menu && (
          <motion.div
            key={`${menu.x}:${menu.y}`}
            ref={menuRef}
            initial={{ opacity: 0, scale: 0.95, y: -4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, transition: { duration: 0.08 } }}
            transition={{ duration: 0.12, ease: "easeOut" }}
            className="absolute z-30 min-w-44 rounded-lg border border-border/70 bg-popover p-1 text-popover-foreground shadow-xl"
            style={{
              insetInlineStart: menu.x,
              top: menu.y,
              transformOrigin: document.documentElement.dir === "rtl" ? "top right" : "top left",
            }}
          >
            {menu.target.type === "pane" && (
              <>
                {ADD_KINDS.map(({ kind, icon, labelKey }) => (
                  <MenuItem
                    key={kind}
                    icon={icon}
                    label={msg(labelKey)}
                    onClick={() => {
                      addNodeAt(kind, menu.flow, menu.pending);
                      setMenu(null);
                    }}
                  />
                ))}
                {!menu.pending && (
                  <>
                    <div className="mx-1 my-1 h-px bg-border/70" />
                    <MenuItem
                      icon={GridFour}
                      label={msg("workflow.toolbar.tidy")}
                      onClick={() => {
                        tidyUp();
                        setMenu(null);
                      }}
                    />
                  </>
                )}
              </>
            )}
            {menuNode && (
              <>
                <MenuItem
                  icon={Copy}
                  label={msg("workflow.menu.duplicate")}
                  onClick={() => {
                    duplicateNode(menuNode);
                    setMenu(null);
                  }}
                />
                <div className="mx-1 my-1 h-px bg-border/70" />
                <MenuItem
                  icon={Trash}
                  label={msg("workflow.menu.delete")}
                  danger
                  onClick={() => {
                    deleteNode(menuNode);
                    setMenu(null);
                  }}
                />
              </>
            )}
            {menuEdge && (
              <MenuItem
                icon={Trash}
                label={msg("workflow.menu.delete_edge")}
                danger
                onClick={() => {
                  deleteEdge(menuEdge);
                  setMenu(null);
                }}
              />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  if (!fullscreen) return body;
  return (
    <>
      {/* Placeholder keeps the wizard card's height while the editor lives
          in the portal, so the page doesn't collapse behind the overlay. */}
      <div className="h-[521px] bg-muted/20" aria-hidden />
      {createPortal(body, document.body)}
    </>
  );
}

function MenuItem({
  icon: Icon,
  label,
  onClick,
  danger = false,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex min-h-[44px] w-full cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-xs transition-colors lg:min-h-0",
        danger ? "text-destructive hover:bg-destructive/10" : "text-foreground hover:bg-muted",
      )}
    >
      <Icon className="size-3.5 shrink-0" />
      <span dir="auto">{label}</span>
    </button>
  );
}

function ZoomControls() {
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
          className="h-[44px] min-w-[44px] cursor-pointer px-1 text-center text-[0.6875rem] font-medium tabular-nums text-muted-foreground transition-colors hover:text-foreground lg:h-7 lg:min-w-12"
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
          // Back to the original framing — the same fit the canvas opened
          // with (mirrors the trajectory tree's reset control).
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
      className="flex size-[44px] cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:bg-muted hover:text-foreground lg:size-7"
    >
      <Icon className="size-3.5" />
    </button>
  );
}

function ToolbarButton({
  icon: Icon,
  label,
  onClick,
  iconOnly = false,
  active = false,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick: () => void;
  iconOnly?: boolean;
  active?: boolean;
}) {
  return (
    <Button
      size="sm"
      variant="ghost"
      className={cn(
        "h-[44px] gap-1.5 px-2 text-xs lg:h-7",
        active ? "bg-[#3D2E22]/8 text-foreground" : "text-muted-foreground hover:text-foreground",
      )}
      onClick={onClick}
      title={iconOnly ? label : undefined}
      aria-label={label}
      aria-pressed={active}
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
