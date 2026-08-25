/**
 * Localizes structural workflow issues (`validateWorkflowSpec` keys) into
 * user-facing copy. Split from `model.ts` so the graph logic stays free of
 * UI imports, and from the canvas so the wizard hook can validate without
 * pulling React Flow into its bundle.
 */

import { formatMsg } from "@/shared/lib/messages";
import type { MessageKey } from "@/shared/lib/messages";
import type { WorkflowIssueKey } from "./model";

export function workflowIssueText(
  key: WorkflowIssueKey,
  params?: Record<string, string>,
): string {
  return formatMsg(`workflow.issue.${key}` as MessageKey, {
    p1: params?.name ?? params?.port ?? params?.params ?? "",
    p2: params?.fields ?? "",
  });
}
