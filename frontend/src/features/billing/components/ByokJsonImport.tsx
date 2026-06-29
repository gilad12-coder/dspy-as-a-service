"use client";

import * as React from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { toast } from "react-toastify";
import { msg, formatMsg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/primitives/button";
import { useByokKeys } from "../providers/byok-provider";

/** One connection entry parsed from the pasted JSON. */
interface ParsedConnection {
  provider: string;
  api_key: string;
  api_base?: string;
  label?: string;
  params?: Record<string, unknown>;
}

/**
 * Parse the textarea into connection entries.
 *
 * Accepts a JSON array (or a single object) of `{ provider, api_key }` with
 * optional `api_base` / `label` / `params`. Returns `null` when the JSON is
 * malformed or no valid entry is present — the caller surfaces one honest error
 * pointing at the required shape.
 */
function parseConnections(raw: string): ParsedConnection[] | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  const list = Array.isArray(data) ? data : [data];
  const out: ParsedConnection[] = [];
  for (const entry of list) {
    if (typeof entry !== "object" || entry === null) return null;
    const e = entry as Record<string, unknown>;
    const provider = typeof e.provider === "string" ? e.provider.trim() : "";
    const apiKey = typeof e.api_key === "string" ? e.api_key.trim() : "";
    if (!provider || !apiKey) return null;
    out.push({
      provider,
      api_key: apiKey,
      api_base: typeof e.api_base === "string" ? e.api_base : undefined,
      label: typeof e.label === "string" ? e.label : undefined,
      params:
        typeof e.params === "object" && e.params !== null
          ? (e.params as Record<string, unknown>)
          : undefined,
    });
  }
  return out.length > 0 ? out : null;
}

/**
 * Advanced JSON import for BYOK connections.
 *
 * A collapsed-by-default disclosure under the provider rows: paste a JSON array
 * of connections to add several at once. Each entry round-trips through the same
 * vault `saveKey` as the manual form (encrypt-at-rest + verify on entry), so the
 * JSON path produces identical records — it never bypasses the front door, it
 * just feeds it in bulk.
 */
export function ByokJsonImport() {
  const { saveKey } = useByokKeys();
  const [open, setOpen] = React.useState(false);
  const [text, setText] = React.useState("");
  const [busy, setBusy] = React.useState(false);

  const handleImport = async () => {
    const parsed = parseConnections(text);
    if (parsed === null) {
      toast.error(msg("settings.keys.json_error"));
      return;
    }
    setBusy(true);
    let imported = 0;
    try {
      for (const c of parsed) {
        await saveKey(c.provider, c.api_key, {
          apiBase: c.api_base ?? null,
          label: c.label ?? null,
          params: c.params,
        });
        imported += 1;
      }
      toast.success(formatMsg("settings.keys.json_imported", { count: imported }));
      setText("");
      setOpen(false);
    } catch {
      // Surface the same shape-guidance error; the rows that did save remain.
      toast.error(msg("settings.keys.json_error"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="border-t border-border/40 pt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        {msg("settings.keys.json_advanced")}
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="mt-2.5 flex flex-col gap-2 animate-in fade-in-0 slide-in-from-top-1">
          <p className="text-[0.6875rem] text-muted-foreground/70">{msg("settings.keys.json_hint")}</p>
          <textarea
            dir="ltr"
            spellCheck={false}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={'[\n  { "provider": "openai", "api_key": "sk-…" }\n]'}
            className="h-32 w-full resize-y rounded-md border border-border/50 bg-background px-2.5 py-2 font-mono text-xs text-foreground placeholder:text-muted-foreground/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
          <div className="flex justify-end">
            <Button size="xs" onClick={handleImport} disabled={!text.trim() || busy}>
              {busy ? <Loader2 className="size-3.5 animate-spin" /> : msg("settings.keys.json_import")}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
