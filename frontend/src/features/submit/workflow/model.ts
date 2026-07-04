/**
 * Workflow graph model helpers for the canvas builder.
 *
 * The wizard owns a `WorkflowSpec` (the exact wire shape the backend
 * validates); this module derives everything the canvas needs from it —
 * per-node ports, structural validation issues, default graphs seeded from
 * the dataset's column roles — and keeps id/field-name rules aligned with
 * the backend (`core/models/workflow.py`).
 */

import type {
  WorkflowEdgeSpec,
  WorkflowFieldSpec,
  WorkflowNodeSpec,
  WorkflowSpec,
} from "@/shared/types/api";
import { buildSignatureTemplate } from "../lib/build-signature";

export const IDENTIFIER_RE = /^[A-Za-z][A-Za-z0-9_]*$/;

export interface PortInfo {
  name: string;
  annotation: string;
}

export interface NodePorts {
  inputs: PortInfo[];
  outputs: PortInfo[];
}

/** A structural problem anchored to a node (or the whole graph when nodeId is null). */
export interface WorkflowIssue {
  nodeId: string | null;
  message: string;
}

// The `: type` annotation is optional — DSPy defaults untyped fields to str,
// and agent-authored signatures routinely omit it.
const SIGNATURE_FIELD_RE =
  /^\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*([^=]+?))?\s*=\s*dspy\.(InputField|OutputField)\s*\(/;

/**
 * Parse a dspy.Signature source into its input/output ports.
 *
 * A lightweight line-regex mirror of the server-side introspection — enough
 * for live port rendering; the backend's exec-based pass remains the source
 * of truth at validation/submit time.
 */
export function parseSignaturePorts(code: string): NodePorts {
  const inputs: PortInfo[] = [];
  const outputs: PortInfo[] = [];
  for (const line of code.split("\n")) {
    const m = SIGNATURE_FIELD_RE.exec(line);
    if (!m) continue;
    const [, name, annotation, kind] = m;
    if (!name || !kind) continue;
    const port = { name, annotation: annotation?.trim() ?? "str" };
    if (kind === "InputField") inputs.push(port);
    else outputs.push(port);
  }
  return { inputs, outputs };
}

const TRANSFORM_DEF_RE = /def\s+transform\s*\(([^)]*)\)/;

/** Parse the transform function's parameter names (empty when unparseable). */
export function parseTransformParams(code: string): string[] {
  const m = TRANSFORM_DEF_RE.exec(code);
  if (!m) return [];
  return (m[1] ?? "")
    .split(",")
    .map((p) => (p.split(/[:=]/)[0] ?? "").trim())
    .filter((p) => p && p !== "self");
}

function toPort(field: WorkflowFieldSpec): PortInfo {
  return { name: field.name, annotation: field.annotation ?? "str" };
}

/** Derive a node's connectable ports from its spec. */
export function nodePorts(node: WorkflowNodeSpec): NodePorts {
  switch (node.kind) {
    case "input":
      return { inputs: [], outputs: node.fields.map(toPort) };
    case "output":
      return { inputs: node.fields.map(toPort), outputs: [] };
    case "signature":
      return parseSignaturePorts(node.signature_code);
    case "transform":
      return { inputs: node.input_fields.map(toPort), outputs: node.output_fields.map(toPort) };
    case "mcp":
      return { inputs: node.input_fields.map(toPort), outputs: [toPort(node.output_field)] };
  }
}

/** True when the node consumes the run-level tool roster (react/mcp). */
export function isToolUserNode(node: WorkflowNodeSpec): boolean {
  return node.kind === "mcp" || (node.kind === "signature" && node.module_name === "react");
}

/** True when any node in the spec needs a tool_source at submit. */
export function workflowUsesTools(spec: WorkflowSpec): boolean {
  return spec.nodes.some(isToolUserNode);
}

/** Generate a fresh node id with the given prefix, unique within the spec. */
export function makeNodeId(prefix: string, spec: WorkflowSpec): string {
  const taken = new Set(spec.nodes.map((n) => n.id));
  let i = 1;
  while (taken.has(`${prefix}_${i}`)) i += 1;
  return `${prefix}_${i}`;
}

/** Would adding source→target close a directed cycle over the current edges? */
export function wouldCreateCycle(spec: WorkflowSpec, source: string, target: string): boolean {
  if (source === target) return true;
  const adjacency = new Map<string, string[]>();
  for (const e of spec.edges) {
    const list = adjacency.get(e.source) ?? [];
    list.push(e.target);
    adjacency.set(e.source, list);
  }
  // Cycle iff source is already reachable from target.
  const seen = new Set<string>();
  const stack = [target];
  while (stack.length) {
    const cur = stack.pop()!;
    if (cur === source) return true;
    if (seen.has(cur)) continue;
    seen.add(cur);
    for (const next of adjacency.get(cur) ?? []) stack.push(next);
  }
  return false;
}

const sanitizeIdentifier = (s: string) =>
  s.replace(/[^a-zA-Z0-9_]/g, "_").replace(/^(\d)/, "_$1") || "field";

/**
 * Build the starter graph for a fresh workflow: input anchor (dataset input
 * columns) → one predict signature node templated from the roles → output
 * anchor (output columns), fully wired since the template reuses the same
 * sanitized field names on both sides.
 */
export function defaultWorkflowSpec(
  roles: Record<string, string>,
  kinds: Record<string, "text" | "image"> = {},
): WorkflowSpec {
  const inputCols = Object.entries(roles)
    .filter(([, r]) => r === "input")
    .map(([c]) => c);
  const outputCols = Object.entries(roles)
    .filter(([, r]) => r === "output")
    .map(([c]) => c);
  const inputFields = (inputCols.length ? inputCols : ["input_field"]).map((c) => ({
    name: sanitizeIdentifier(c),
  }));
  const outputFields = (outputCols.length ? outputCols : ["output_field"]).map((c) => ({
    name: sanitizeIdentifier(c),
  }));

  const nodes: WorkflowNodeSpec[] = [
    { id: "input", kind: "input", fields: inputFields, position: { x: 0, y: 120 } },
    {
      id: "step_1",
      kind: "signature",
      module_name: "predict",
      signature_code: buildSignatureTemplate(roles, kinds),
      position: { x: 320, y: 80 },
    },
    { id: "output", kind: "output", fields: outputFields, position: { x: 680, y: 120 } },
  ];
  const edges: WorkflowEdgeSpec[] = [
    ...inputFields.map((f) => ({
      source: "input",
      source_port: f.name,
      target: "step_1",
      target_port: f.name,
    })),
    ...outputFields.map((f) => ({
      source: "step_1",
      source_port: f.name,
      target: "output",
      target_port: f.name,
    })),
  ];
  return { nodes, edges };
}

/** Template specs for nodes added from the canvas toolbar. */
export function newNodeSpec(
  kind: "signature" | "transform" | "mcp",
  spec: WorkflowSpec,
  position: { x: number; y: number },
): WorkflowNodeSpec {
  switch (kind) {
    case "signature":
      return {
        id: makeNodeId("step", spec),
        kind: "signature",
        module_name: "predict",
        signature_code:
          'class Step(dspy.Signature):\n    """Describe this step."""\n\n    text: str = dspy.InputField(desc="")\n    result: str = dspy.OutputField(desc="")\n',
        position,
      };
    case "transform":
      return {
        id: makeNodeId("transform", spec),
        kind: "transform",
        transform_code: "def transform(text):\n    return {\"result\": text}\n",
        input_fields: [{ name: "text" }],
        output_fields: [{ name: "result" }],
        position,
      };
    case "mcp":
      return {
        id: makeNodeId("tool", spec),
        kind: "mcp",
        tool_name: "",
        input_fields: [{ name: "query" }],
        output_field: { name: "result" },
        position,
      };
  }
}

/**
 * Layered auto-layout: column = longest-path depth from the input anchor,
 * row = order within the column. Deterministic and dependency-free; used by
 * the canvas tidy-up button and to place agent-authored nodes (which arrive
 * without positions).
 */
export function autoLayoutSpec(spec: WorkflowSpec): WorkflowSpec {
  const depth = new Map<string, number>(spec.nodes.map((n) => [n.id, 0]));
  for (let pass = 0; pass < spec.nodes.length; pass += 1) {
    let changed = false;
    for (const e of spec.edges) {
      const next = (depth.get(e.source) ?? 0) + 1;
      if (next > (depth.get(e.target) ?? 0)) {
        depth.set(e.target, next);
        changed = true;
      }
    }
    if (!changed) break;
  }
  const output = spec.nodes.find((n) => n.kind === "output");
  if (output && depth.size > 0) {
    depth.set(output.id, Math.max(...depth.values()));
  }
  const rows = new Map<number, number>();
  return {
    ...spec,
    nodes: spec.nodes.map((n) => {
      const d = depth.get(n.id) ?? 0;
      const row = rows.get(d) ?? 0;
      rows.set(d, row + 1);
      return { ...n, position: { x: d * 300, y: row * 160 + (d % 2) * 40 } };
    }),
  };
}

/**
 * Client-side structural validation, mirroring the backend's exec-free pass.
 *
 * Messages are pre-localized by the caller-supplied `t` so this module stays
 * free of UI imports; each issue anchors to a node for canvas badges.
 */
export function validateWorkflowSpec(
  spec: WorkflowSpec,
  t: (key: WorkflowIssueKey, params?: Record<string, string>) => string,
): WorkflowIssue[] {
  const issues: WorkflowIssue[] = [];
  const byId = new Map(spec.nodes.map((n) => [n.id, n]));
  const ports = new Map(spec.nodes.map((n) => [n.id, nodePorts(n)]));

  for (const node of spec.nodes) {
    const p = ports.get(node.id)!;
    const declared = [...p.inputs, ...p.outputs];
    for (const port of declared) {
      if (!IDENTIFIER_RE.test(port.name)) {
        issues.push({ nodeId: node.id, message: t("bad_field_name", { name: port.name }) });
      }
    }
    const seen = new Set<string>();
    for (const port of p.inputs) {
      if (seen.has(port.name))
        issues.push({ nodeId: node.id, message: t("duplicate_field", { name: port.name }) });
      seen.add(port.name);
    }
    if (node.kind === "signature") {
      if (!node.signature_code.trim() || (p.inputs.length === 0 && p.outputs.length === 0)) {
        issues.push({ nodeId: node.id, message: t("signature_empty") });
      }
    }
    if (node.kind === "mcp" && !node.tool_name.trim()) {
      issues.push({ nodeId: node.id, message: t("tool_name_missing") });
    }
    if (node.kind === "transform") {
      const params = [...parseTransformParams(node.transform_code)].sort();
      const declaredInputs = node.input_fields.map((f) => f.name).sort();
      if (params.length && JSON.stringify(params) !== JSON.stringify(declaredInputs)) {
        issues.push({
          nodeId: node.id,
          message: t("transform_params_mismatch", {
            params: params.join(", "),
            fields: declaredInputs.join(", "),
          }),
        });
      }
    }
  }

  const fed = new Set(spec.edges.map((e) => `${e.target} ${e.target_port}`));
  const feedCounts = new Map<string, number>();
  for (const e of spec.edges) {
    const key = `${e.target} ${e.target_port}`;
    feedCounts.set(key, (feedCounts.get(key) ?? 0) + 1);
  }
  for (const [key, count] of feedCounts) {
    if (count > 1) {
      const [nodeId = "", port = ""] = key.split(" ");
      issues.push({ nodeId, message: t("multi_producer", { port }) });
    }
  }

  for (const node of spec.nodes) {
    if (node.kind === "input") continue;
    const required = ports.get(node.id)!.inputs;
    const unfed = required.filter((port) => !fed.has(`${node.id} ${port.name}`));
    for (const port of unfed) {
      issues.push({ nodeId: node.id, message: t("unconnected_input", { port: port.name }) });
    }
  }

  for (const e of spec.edges) {
    const sourcePorts = ports.get(e.source);
    const targetPorts = ports.get(e.target);
    if (!sourcePorts || !targetPorts) continue;
    if (!sourcePorts.outputs.some((p) => p.name === e.source_port)) {
      issues.push({ nodeId: e.source, message: t("missing_port", { port: e.source_port }) });
    }
    if (!targetPorts.inputs.some((p) => p.name === e.target_port)) {
      issues.push({ nodeId: e.target, message: t("missing_port", { port: e.target_port }) });
    }
  }

  // Reachability: every node must sit on some input→output path.
  const input = spec.nodes.find((n) => n.kind === "input");
  const output = spec.nodes.find((n) => n.kind === "output");
  if (input && output) {
    const forward = new Map<string, string[]>();
    const backward = new Map<string, string[]>();
    for (const e of spec.edges) {
      forward.set(e.source, [...(forward.get(e.source) ?? []), e.target]);
      backward.set(e.target, [...(backward.get(e.target) ?? []), e.source]);
    }
    const flood = (adj: Map<string, string[]>, start: string) => {
      const seen = new Set<string>();
      const stack = [...(adj.get(start) ?? [])];
      while (stack.length) {
        const cur = stack.pop()!;
        if (seen.has(cur)) continue;
        seen.add(cur);
        stack.push(...(adj.get(cur) ?? []));
      }
      return seen;
    };
    const reachable = flood(forward, input.id);
    const coReachable = flood(backward, output.id);
    for (const node of spec.nodes) {
      if (node.id !== input.id && !reachable.has(node.id)) {
        issues.push({ nodeId: node.id, message: t("unreachable") });
      } else if (node.id !== output.id && !coReachable.has(node.id)) {
        issues.push({ nodeId: node.id, message: t("no_path_to_output") });
      }
    }
  }

  // Cycle check via Kahn over unique node pairs.
  const indegree = new Map(spec.nodes.map((n) => [n.id, 0]));
  const outgoing = new Map<string, Set<string>>(spec.nodes.map((n) => [n.id, new Set()]));
  for (const e of spec.edges) {
    if (!byId.has(e.source) || !byId.has(e.target)) continue;
    if (!outgoing.get(e.source)!.has(e.target)) {
      outgoing.get(e.source)!.add(e.target);
      indegree.set(e.target, (indegree.get(e.target) ?? 0) + 1);
    }
  }
  const ready = spec.nodes.filter((n) => (indegree.get(n.id) ?? 0) === 0).map((n) => n.id);
  let visited = 0;
  while (ready.length) {
    const id = ready.pop()!;
    visited += 1;
    for (const child of outgoing.get(id) ?? []) {
      indegree.set(child, indegree.get(child)! - 1);
      if (indegree.get(child) === 0) ready.push(child);
    }
  }
  if (visited !== spec.nodes.length) {
    issues.push({ nodeId: null, message: t("cycle") });
  }

  return issues;
}

export type WorkflowIssueKey =
  | "bad_field_name"
  | "duplicate_field"
  | "signature_empty"
  | "tool_name_missing"
  | "transform_params_mismatch"
  | "multi_producer"
  | "unconnected_input"
  | "missing_port"
  | "unreachable"
  | "no_path_to_output"
  | "cycle";
