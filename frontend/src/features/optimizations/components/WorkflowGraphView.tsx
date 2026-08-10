"use client";

import { useMemo } from "react";
import {
  Background,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { autoLayoutSpec } from "@/features/submit/workflow/model";
import { NODE_TYPES, flowTypeFor, type CanvasNode } from "@/features/submit/workflow/nodes";
import type { WorkflowSpec } from "@/shared/types/api";

/** Read-only rendering of a run's workflow graph for the detail page. */
export function WorkflowGraphView({ spec }: { spec: WorkflowSpec }) {
  const { nodes, edges } = useMemo(() => {
    // Runs submitted through the canvas carry positions; auto-layout covers
    // programmatically submitted specs that ship without them.
    const laid = spec.nodes.every((n) => n.position) ? spec : autoLayoutSpec(spec);
    const flowNodes: CanvasNode[] = laid.nodes.map((node) => ({
      id: node.id,
      type: flowTypeFor(node),
      position: node.position ?? { x: 0, y: 0 },
      data: { spec: node, issues: [], trace: null, pulse: false },
    }));
    const flowEdges: Edge[] = laid.edges.map((e) => ({
      id: `${e.source}.${e.source_port}->${e.target}.${e.target_port}`,
      source: e.source,
      target: e.target,
      sourceHandle: e.source_port,
      targetHandle: e.target_port,
      style: { stroke: "#8A7563", strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#8A7563", width: 16, height: 16 },
    }));
    return { nodes: flowNodes, edges: flowEdges };
  }, [spec]);

  return (
    <div dir="ltr" className="h-[480px] overflow-hidden rounded-lg border border-border/60">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          edgesFocusable={false}
          zoomOnDoubleClick={false}
          minZoom={0.1}
          fitView
          fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
          proOptions={{ hideAttribution: true }}
          className="bg-[#FDFCFA]"
        >
          <Background gap={16} size={1.25} color="#E3D9CB" />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}

export default WorkflowGraphView;
