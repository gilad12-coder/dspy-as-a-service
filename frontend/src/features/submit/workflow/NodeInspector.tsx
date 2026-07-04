"use client";

/**
 * Inspector panel for the workflow canvas: edits the selected node's spec.
 *
 * Purely controlled — receives the node spec and emits a replacement via
 * `onChange`; the canvas owns spec state and re-derives ports/edges. Field
 * renames intentionally do NOT rewrite existing edges: the stale edge shows
 * up as a validation issue on the node, which is easier to reason about than
 * silent rewiring.
 */

import * as React from "react";
import dynamic from "next/dynamic";
import { Plus, Trash2, X } from "lucide-react";

import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { Label } from "@/shared/ui/primitives/label";
import { Separator } from "@/shared/ui/primitives/separator";
import { Skeleton } from "@/shared/ui/skeleton";
import { cn } from "@/shared/lib/utils";
import { msg } from "@/shared/lib/messages";
import type { WorkflowFieldSpec, WorkflowNodeSpec } from "@/shared/types/api";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={160} borderRadius={8} />,
});

const MODULE_CHOICES = [
  ["predict", "Predict"],
  ["cot", "CoT"],
  ["react", "ReAct"],
] as const;

interface NodeInspectorProps {
  spec: WorkflowNodeSpec;
  issues: string[];
  onChange: (next: WorkflowNodeSpec) => void;
  onDelete?: () => void;
  onClose?: () => void;
}

export function NodeInspector({ spec, issues, onChange, onDelete, onClose }: NodeInspectorProps) {
  return (
    <div className="flex h-full flex-col overflow-y-auto bg-card">
      <div className="flex items-center justify-between gap-2 border-b border-border/60 px-4 py-2.5">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold" dir="ltr">
            {spec.id}
          </div>
          <div className="text-[0.6875rem] text-muted-foreground">
            {msg(`workflow.inspector.kind.${spec.kind}`)}
          </div>
        </div>
        <div className="flex shrink-0 items-center">
          {onDelete && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onDelete}
              className="text-muted-foreground hover:text-destructive"
              aria-label={msg("workflow.inspector.delete")}
            >
              <Trash2 className="size-3.5" />
            </Button>
          )}
          {onClose && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground"
              aria-label={msg("workflow.inspector.close")}
            >
              <X className="size-3.5" />
            </Button>
          )}
        </div>
      </div>

      {issues.length > 0 && (
        <div className="border-b border-border/60 bg-[#FBF3EC] px-4 py-2">
          <ul className="list-disc space-y-0.5 ps-4 text-[0.6875rem] leading-relaxed text-[#A3512B]">
            {issues.map((issue, i) => (
              <li key={i}>{issue}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="space-y-4 px-4 py-3">
        <div className="space-y-1.5">
          <Label className="text-[0.6875rem] uppercase tracking-wide text-muted-foreground">
            {msg("workflow.inspector.display_name")}
          </Label>
          <Input
            value={spec.name ?? ""}
            placeholder={spec.id}
            onChange={(e) => onChange({ ...spec, name: e.target.value || null })}
          />
        </div>

        {(spec.kind === "input" || spec.kind === "output") && (
          <FieldListEditor
            label={msg(
              spec.kind === "input"
                ? "workflow.inspector.input_fields"
                : "workflow.inspector.output_fields",
            )}
            fields={spec.fields}
            minFields={1}
            onChange={(fields) => onChange({ ...spec, fields })}
          />
        )}

        {spec.kind === "signature" && (
          <>
            <div className="space-y-1.5">
              <Label className="text-[0.6875rem] uppercase tracking-wide text-muted-foreground">
                {msg("workflow.inspector.module")}
              </Label>
              <div className="inline-flex w-full rounded-lg bg-muted p-1 gap-1">
                {MODULE_CHOICES.map(([val, label]) => (
                  <button
                    key={val}
                    type="button"
                    onClick={() =>
                      onChange({
                        ...spec,
                        module_name: val,
                        ...(val !== "react" ? { tool_filter: null } : {}),
                      })
                    }
                    className={cn(
                      "flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors cursor-pointer",
                      spec.module_name === val
                        ? "bg-background text-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-1.5">
              <Label className="text-[0.6875rem] uppercase tracking-wide text-muted-foreground">
                {msg("workflow.inspector.signature_code")}
              </Label>
              <CodeEditor
                value={spec.signature_code}
                onChange={(v) => onChange({ ...spec, signature_code: v })}
                height="220px"
              />
            </div>
          </>
        )}

        {spec.kind === "transform" && (
          <>
            <FieldListEditor
              label={msg("workflow.inspector.input_fields")}
              fields={spec.input_fields}
              minFields={1}
              onChange={(input_fields) => onChange({ ...spec, input_fields })}
            />
            <FieldListEditor
              label={msg("workflow.inspector.output_fields")}
              fields={spec.output_fields}
              minFields={1}
              onChange={(output_fields) => onChange({ ...spec, output_fields })}
            />
            <div className="space-y-1.5">
              <Label className="text-[0.6875rem] uppercase tracking-wide text-muted-foreground">
                {msg("workflow.inspector.transform_code")}
              </Label>
              <CodeEditor
                value={spec.transform_code}
                onChange={(v) => onChange({ ...spec, transform_code: v })}
                height="180px"
              />
            </div>
          </>
        )}

        {spec.kind === "mcp" && (
          <>
            <div className="space-y-1.5">
              <Label className="text-[0.6875rem] uppercase tracking-wide text-muted-foreground">
                {msg("workflow.inspector.tool_name")}
              </Label>
              <Input
                dir="ltr"
                value={spec.tool_name}
                placeholder={msg("workflow.inspector.tool_name_placeholder")}
                onChange={(e) => onChange({ ...spec, tool_name: e.target.value })}
              />
            </div>
            <FieldListEditor
              label={msg("workflow.inspector.input_fields")}
              fields={spec.input_fields}
              minFields={0}
              onChange={(input_fields) => onChange({ ...spec, input_fields })}
            />
            <div className="space-y-1.5">
              <Label className="text-[0.6875rem] uppercase tracking-wide text-muted-foreground">
                {msg("workflow.inspector.result_field")}
              </Label>
              <Input
                dir="ltr"
                value={spec.output_field.name}
                onChange={(e) =>
                  onChange({
                    ...spec,
                    output_field: { ...spec.output_field, name: e.target.value },
                  })
                }
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function FieldListEditor({
  label,
  fields,
  minFields,
  onChange,
}: {
  label: string;
  fields: WorkflowFieldSpec[];
  minFields: number;
  onChange: (fields: WorkflowFieldSpec[]) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <Label className="text-[0.6875rem] uppercase tracking-wide text-muted-foreground">
          {label}
        </Label>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-1.5 text-muted-foreground"
          onClick={() => onChange([...fields, { name: `field_${fields.length + 1}` }])}
          aria-label={msg("workflow.inspector.add_field")}
        >
          <Plus className="size-3.5" />
        </Button>
      </div>
      <div className="space-y-1">
        {fields.map((field, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <Input
              dir="ltr"
              className="h-8 font-mono text-xs"
              value={field.name}
              onChange={(e) => {
                const next = [...fields];
                next[i] = { ...field, name: e.target.value };
                onChange(next);
              }}
            />
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-1.5 text-muted-foreground hover:text-destructive"
              disabled={fields.length <= minFields}
              onClick={() => onChange(fields.filter((_, j) => j !== i))}
              aria-label={msg("workflow.inspector.remove_field")}
            >
              <Trash2 className="size-3" />
            </Button>
          </div>
        ))}
      </div>
      <Separator className="mt-2" />
    </div>
  );
}
