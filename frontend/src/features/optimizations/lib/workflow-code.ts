/**
 * Compile a workflow graph spec into readable DSPy Python source.
 *
 * Display-only rendering for the detail page's Code tab: signature nodes
 * appear verbatim (their docstrings are the initial prompts GEPA optimizes),
 * transform code is inlined as functions named after their nodes, and a
 * `WorkflowProgram` module mirrors how the backend builds the graph —
 * sub-modules attached as `n_<node_id>`, executed in dependency order with
 * edges wired port to port.
 */

import type {
  WorkflowNodeSpec,
  WorkflowSignatureNodeSpec,
  WorkflowSpec,
  WorkflowTransformNodeSpec,
} from "@/shared/types/api";

const MODULE_CLASSES = {
  predict: "dspy.Predict",
  cot: "dspy.ChainOfThought",
  react: "dspy.ReAct",
  flex: "dspy.Flex",
} as const;

const SIGNATURE_CLASS_RE = /class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/;
const TRANSFORM_DEF_RE = /def\s+transform\s*\(/;
const FIRST_DEF_RE = /def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/;
const IMPORT_LINE_RE = /^(?:import\s+\S|from\s+\S+\s+import\s)/;

const MAX_CALL_LINE = 100;

const sanitizeIdentifier = (raw: string): string =>
  raw.replace(/[^A-Za-z0-9_]/g, "_").replace(/^(\d)/, "_$1") || "tool";

/**
 * Kahn topological order with ties broken by spec order, matching the
 * backend's `workflow_topological_order` so the generated forward() runs
 * nodes in the same order the platform does.
 */
function topologicalOrder(spec: WorkflowSpec): string[] {
  const indegree = new Map(spec.nodes.map((n) => [n.id, 0]));
  const outgoing = new Map<string, Set<string>>(spec.nodes.map((n) => [n.id, new Set()]));
  for (const e of spec.edges) {
    const out = outgoing.get(e.source);
    if (!out || !indegree.has(e.target) || out.has(e.target)) continue;
    out.add(e.target);
    indegree.set(e.target, (indegree.get(e.target) ?? 0) + 1);
  }
  const ordered: string[] = [];
  const emitted = new Set<string>();
  for (;;) {
    const next = spec.nodes.find((n) => !emitted.has(n.id) && (indegree.get(n.id) ?? 0) === 0);
    if (!next) break;
    emitted.add(next.id);
    ordered.push(next.id);
    for (const child of outgoing.get(next.id) ?? []) {
      indegree.set(child, (indegree.get(child) ?? 0) - 1);
    }
  }
  // Cycles can't pass submission validation; if one slips in, still render
  // every node rather than dropping the tail.
  for (const n of spec.nodes) if (!emitted.has(n.id)) ordered.push(n.id);
  return ordered;
}

interface LeadingImportSplit {
  imports: string[];
  body: string;
}

/**
 * Split the import lines a snippet starts with from its body, so node code
 * submitted with its own `import dspy` etc. can share one hoisted header
 * instead of repeating imports per node.
 */
function splitLeadingImports(code: string): LeadingImportSplit {
  const lines = code.split("\n");
  const imports: string[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = (lines[i] ?? "").trim();
    if (line !== "" && !IMPORT_LINE_RE.test(line)) break;
    if (line) imports.push(line);
    i += 1;
  }
  return { imports, body: lines.slice(i).join("\n") };
}

interface SignatureRender {
  code: string;
  className: string;
}

/**
 * Extract (and de-collide) the signature class a node's code defines.
 *
 * Two nodes may both define `class Step(...)`; the second gets suffixed with
 * its node id so the generated module references stay unambiguous.
 */
function renderSignature(node: WorkflowSignatureNodeSpec, used: Set<string>): SignatureRender {
  let code = node.signature_code.trimEnd();
  const match = SIGNATURE_CLASS_RE.exec(code);
  let className = match?.[1] ?? `Signature_${node.id}`;
  if (used.has(className)) {
    const unique = sanitizeIdentifier(`${className}_${node.id}`);
    if (match) code = code.replace(match[0], `class ${unique}(`);
    className = unique;
  }
  used.add(className);
  return { code, className };
}

interface TransformRender {
  code: string;
  fnName: string;
}

/** Rename the conventional `def transform(...)` to the node id for uniqueness. */
function renderTransform(node: WorkflowTransformNodeSpec): TransformRender {
  const code = node.transform_code.trimEnd();
  if (TRANSFORM_DEF_RE.test(code)) {
    return { code: code.replace(TRANSFORM_DEF_RE, `def ${node.id}(`), fnName: node.id };
  }
  const match = FIRST_DEF_RE.exec(code);
  return { code, fnName: match?.[1] ?? node.id };
}

/** Format `target = callee(kw=..., ...)`, wrapping when the line runs long. */
function renderCall(
  indent: string,
  target: string,
  callee: string,
  kwargs: Array<[string, string]>,
  comment?: string,
): string[] {
  const inline = kwargs.map(([k, v]) => `${k}=${v}`).join(", ");
  const suffix = comment ? `  # ${comment}` : "";
  const single = `${indent}${target}${callee}(${inline})${suffix}`;
  if (single.length <= MAX_CALL_LINE) return [single];
  return [
    `${indent}${target}${callee}(${suffix}`,
    ...kwargs.map(([k, v]) => `${indent}    ${k}=${v},`),
    `${indent})`,
  ];
}

/** The Python expression that reads `port` off an already-executed node. */
function sourceExpr(node: WorkflowNodeSpec, port: string): string {
  switch (node.kind) {
    case "input":
      return port;
    case "signature":
      return `${node.id}_out.${port}`;
    case "transform":
      return `${node.id}_out["${port}"]`;
    case "mcp":
      return `${node.id}_out`;
    default:
      return port;
  }
}

/** Compile the workflow spec to a readable, standalone DSPy program source. */
export function compileWorkflowToCode(spec: WorkflowSpec): string {
  const byId = new Map(spec.nodes.map((n) => [n.id, n]));
  const order = topologicalOrder(spec);

  const hoistedImports: string[] = [];
  const seenImports = new Set<string>(["import dspy"]);
  const takeImports = (code: string): string => {
    const { imports, body } = splitLeadingImports(code);
    for (const imp of imports) {
      if (!seenImports.has(imp)) {
        seenImports.add(imp);
        hoistedImports.push(imp);
      }
    }
    return body;
  };

  const usedClassNames = new Set<string>();
  const signatures = new Map<string, SignatureRender>();
  const transforms = new Map<string, TransformRender>();
  for (const id of order) {
    const node = byId.get(id);
    if (node?.kind === "signature") {
      const r = renderSignature(node, usedClassNames);
      signatures.set(id, { ...r, code: takeImports(r.code) });
    }
    if (node?.kind === "transform") {
      const r = renderTransform(node);
      transforms.set(id, { ...r, code: takeImports(r.code) });
    }
  }

  // Tool roster: MCP-node tools plus react/flex tool filters, each mapped to
  // a Python identifier the generated calls can reference.
  const toolFns = new Map<string, string>();
  const usedToolFns = new Set<string>();
  const toolFor = (name: string): string => {
    const existing = toolFns.get(name);
    if (existing) return existing;
    let fn = sanitizeIdentifier(name);
    while (usedToolFns.has(fn)) fn = `${fn}_`;
    usedToolFns.add(fn);
    toolFns.set(name, fn);
    return fn;
  };
  for (const node of spec.nodes) {
    if (node.kind === "mcp") toolFor(node.tool_name);
    if (node.kind === "signature") for (const name of node.tool_filter ?? []) toolFor(name);
  }

  const lines: string[] = [
    '"""Workflow program compiled from this run\'s graph spec.',
    "",
    "Auto-generated rendering: each signature class appears exactly as",
    "submitted (its docstring is the initial prompt GEPA optimizes), and",
    "forward() executes the nodes in the platform's dependency order.",
    '"""',
    "",
    "import dspy",
    ...hoistedImports,
  ];

  if (toolFns.size > 0) {
    lines.push("", "# Tools resolve at runtime from the run's MCP tool source, matched by name:");
    for (const [name, fn] of toolFns) {
      lines.push(`#   ${fn}${fn === name ? "" : `  (tool "${name}")`}`);
    }
  }

  for (const id of order) {
    const sig = signatures.get(id);
    if (!sig) continue;
    const node = byId.get(id) as WorkflowSignatureNodeSpec;
    lines.push("", "", `# node: ${id} · module: ${MODULE_CLASSES[node.module_name]}`, sig.code);
  }

  for (const id of order) {
    const t = transforms.get(id);
    if (!t) continue;
    lines.push("", "", `# node: ${id} · transform`, t.code);
  }

  lines.push("", "", "class WorkflowProgram(dspy.Module):");
  lines.push('    """Runs the graph node by node, in dependency order."""', "");
  lines.push("    def __init__(self):");
  lines.push("        super().__init__()");
  for (const id of order) {
    const node = byId.get(id);
    if (node?.kind !== "signature") continue;
    const className = signatures.get(id)!.className;
    const moduleCls = MODULE_CLASSES[node.module_name];
    let toolsArg = "";
    if (node.tool_filter && node.tool_filter.length > 0) {
      toolsArg = `, tools=[${node.tool_filter.map(toolFor).join(", ")}]`;
    } else if (node.module_name === "react") {
      toolsArg = ", tools=TOOLS  # full run-level tool roster";
    }
    lines.push(`        self.n_${id} = ${moduleCls}(${className}${toolsArg})`);
  }

  const inputNode = spec.nodes.find((n) => n.kind === "input");
  const inputFields = inputNode?.kind === "input" ? inputNode.fields.map((f) => f.name) : [];
  lines.push("", `    def forward(self, ${inputFields.join(", ")}):`);

  let emittedBody = false;
  for (const id of order) {
    const node = byId.get(id);
    if (!node || node.kind === "input") continue;
    const incoming = spec.edges.filter((e) => e.target === id);
    const kwargs: Array<[string, string]> = incoming.map((e) => {
      const source = byId.get(e.source);
      return [e.target_port, source ? sourceExpr(source, e.source_port) : e.source_port];
    });
    if (node.kind === "output") {
      lines.push(...renderCall("        ", "return ", "dspy.Prediction", kwargs));
      emittedBody = true;
      continue;
    }
    let callee: string;
    let comment: string | undefined;
    if (node.kind === "signature") {
      callee = `self.n_${id}`;
    } else if (node.kind === "transform") {
      callee = transforms.get(id)!.fnName;
    } else {
      callee = toolFor(node.tool_name);
      if (callee !== node.tool_name) comment = `MCP tool "${node.tool_name}"`;
    }
    lines.push(...renderCall("        ", `${id}_out = `, callee, kwargs, comment));
    emittedBody = true;
  }
  if (!emittedBody) lines.push("        pass");

  return `${lines.join("\n")}\n`;
}
