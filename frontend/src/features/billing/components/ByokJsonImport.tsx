"use client";

import * as React from "react";
import { AlertTriangle, CheckCircle2, ChevronDown, Loader2, ShieldCheck } from "lucide-react";
import { toast } from "react-toastify";
import { msg, formatMsg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/primitives/button";
import { useByokKeys } from "../providers/byok-provider";
import { BYOK_PROVIDERS } from "../lib/byok";
import { ProviderLogo } from "./ProviderLogo";

/** One connection entry parsed and validated from the pasted JSON. */
interface ParsedConnection {
  provider: string;
  api_key: string;
  api_base?: string;
  label?: string;
  params?: Record<string, unknown>;
}

/** Outcome of validating the textarea: the clean rows plus any blocking errors / soft warnings. */
interface Validation {
  ok: boolean;
  connections: ParsedConnection[];
  errors: string[];
  warnings: string[];
}

const OFFERED = BYOK_PROVIDERS.map((p) => p.slug);
const OFFERED_SET = new Set(OFFERED);
const LABEL_FOR: Record<string, string> = Object.fromEntries(
  BYOK_PROVIDERS.map((p) => [p.slug, p.label]),
);

// A worked example users can drop in and adapt — one managed provider, one
// keyed provider, and a custom OpenAI-compatible endpoint (the three shapes).
const EXAMPLE = JSON.stringify(
  [
    { provider: "openai", api_key: "sk-…", label: "My OpenAI key" },
    { provider: "groq", api_key: "gsk_…", label: "Groq personal" },
    {
      provider: "custom",
      api_base: "https://api.example.com/v1",
      api_key: "abc123",
      label: "Local gateway",
    },
  ],
  null,
  2,
);

/** A provider label for display — the offered brand name, or the raw slug for custom/unknown. */
function providerLabel(slug: string): string {
  return LABEL_FOR[slug] ?? slug;
}

/** Show a key's head and mask the rest, so a pasted secret is recognizable but never exposed. */
function maskKey(secret: string): string {
  return secret.length > 4 ? `${secret.slice(0, 4)}••••` : "••••";
}

/** True when a string parses as an http(s) URL — guards the optional api_base. */
function isHttpUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * Parse + validate the textarea into connection rows.
 *
 * Accepts a JSON array (or a single object). Each item is checked for the
 * required `provider`/`api_key`, a provider in our offered set (or `custom`
 * with an `api_base`), and a well-formed `api_base`. Blocking problems land in
 * `errors`; soft ones (a key that will replace an existing connection, a
 * duplicate provider in the same paste) land in `warnings`. The clean rows are
 * collected in `connections` so a paste with one bad item can still preview the
 * good ones (import stays gated on zero errors).
 */
function validate(raw: string, existing: Set<string>): Validation {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return {
      ok: false,
      connections: [],
      errors: [msg("settings.keys.json_err_parse")],
      warnings: [],
    };
  }

  const list = Array.isArray(data) ? data : [data];
  const errors: string[] = [];
  const warnings: string[] = [];
  const connections: ParsedConnection[] = [];
  const seen = new Set<string>();

  list.forEach((entry, index) => {
    const n = index + 1;
    if (typeof entry !== "object" || entry === null) {
      errors.push(formatMsg("settings.keys.json_err_not_object", { n }));
      return;
    }
    const e = entry as Record<string, unknown>;
    const provider = typeof e.provider === "string" ? e.provider.trim() : "";
    const apiKey = typeof e.api_key === "string" ? e.api_key.trim() : "";
    const apiBase = typeof e.api_base === "string" ? e.api_base.trim() : "";
    const label = typeof e.label === "string" ? e.label.trim() : "";

    let itemOk = true;
    if (!provider) {
      errors.push(formatMsg("settings.keys.json_err_missing", { n, field: "provider" }));
      itemOk = false;
    }
    if (!apiKey) {
      errors.push(formatMsg("settings.keys.json_err_missing", { n, field: "api_key" }));
      itemOk = false;
    }
    if (provider && provider !== "custom" && !OFFERED_SET.has(provider)) {
      errors.push(
        formatMsg("settings.keys.json_err_provider", { n, providers: OFFERED.join(", ") }),
      );
      itemOk = false;
    }
    if (provider === "custom" && !apiBase) {
      errors.push(formatMsg("settings.keys.json_err_custom_base", { n }));
      itemOk = false;
    }
    if (apiBase && !isHttpUrl(apiBase)) {
      errors.push(formatMsg("settings.keys.json_err_base", { n }));
      itemOk = false;
    }

    if (provider) {
      if (seen.has(provider)) {
        warnings.push(
          formatMsg("settings.keys.json_warn_dupe", { n, provider: providerLabel(provider) }),
        );
      }
      seen.add(provider);
      if (itemOk && existing.has(provider)) {
        warnings.push(
          formatMsg("settings.keys.json_warn_replace", { provider: providerLabel(provider) }),
        );
      }
    }

    if (itemOk) {
      connections.push({
        provider,
        api_key: apiKey,
        api_base: apiBase || undefined,
        label: label || undefined,
        params:
          typeof e.params === "object" && e.params !== null
            ? (e.params as Record<string, unknown>)
            : undefined,
      });
    }
  });

  return { ok: errors.length === 0 && connections.length > 0, connections, errors, warnings };
}

/**
 * Advanced JSON importer for BYOK connections — a power-user escape hatch.
 *
 * Collapsed by default under the provider rows. Expanded, it's a small import
 * wizard: paste a JSON array, optionally pull in an example or pretty-print it,
 * then Validate to see a per-item error/warning pass and a masked preview table
 * before committing. Import round-trips every row through the same vault
 * `saveKey` as the manual form (encrypt-at-rest + verify on entry), so a saved
 * connection lands in its provider row exactly as a hand-typed one would — the
 * bulk path feeds the front door, it never bypasses it.
 */
export function ByokJsonImport() {
  const { saveKey, keys } = useByokKeys();
  const [open, setOpen] = React.useState(false);
  const [text, setText] = React.useState("");
  const [showResults, setShowResults] = React.useState(false);
  const [busy, setBusy] = React.useState(false);

  const existing = React.useMemo(() => new Set(keys.map((k) => k.provider)), [keys]);
  const result = React.useMemo<Validation | null>(
    () => (text.trim() ? validate(text, existing) : null),
    [text, existing],
  );

  const handleUseExample = () => {
    setText(EXAMPLE);
    setShowResults(false);
  };

  const handleFormat = () => {
    try {
      setText(JSON.stringify(JSON.parse(text), null, 2));
    } catch {
      // Can't pretty-print invalid JSON — reveal the parse error instead.
      setShowResults(true);
    }
  };

  const handleImport = async () => {
    if (!result?.ok) return;
    setBusy(true);
    let imported = 0;
    let failed = 0;
    try {
      for (const c of result.connections) {
        try {
          await saveKey(c.provider, c.api_key, {
            apiBase: c.api_base ?? null,
            label: c.label ?? null,
            params: c.params,
          });
          imported += 1;
        } catch {
          failed += 1;
        }
      }
      if (failed === 0) {
        toast.success(formatMsg("settings.keys.json_imported", { count: imported }));
        setText("");
        setShowResults(false);
        setOpen(false);
      } else {
        toast.warn(
          formatMsg("settings.keys.json_partial", {
            ok: imported,
            total: imported + failed,
            failed,
          }),
        );
      }
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
        className="flex w-full cursor-pointer items-center justify-between text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        {msg("settings.keys.json_advanced")}
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="mt-3 flex flex-col gap-3 animate-in fade-in-0 slide-in-from-top-1">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-semibold text-foreground">
              {msg("settings.keys.json_title")}
            </span>
            <p className="text-[0.6875rem] leading-relaxed text-muted-foreground">
              {msg("settings.keys.json_intro")}
            </p>
          </div>

          <textarea
            dir="ltr"
            spellCheck={false}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={'[\n  { "provider": "openai", "api_key": "sk-…" }\n]'}
            className="h-36 w-full resize-y rounded-md border border-border/50 bg-background px-2.5 py-2 font-mono text-xs text-foreground placeholder:text-muted-foreground/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />

          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-col gap-0.5 text-[0.625rem] text-muted-foreground/70">
              <span>{msg("settings.keys.json_required")}</span>
              <span>{msg("settings.keys.json_optional")}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Button variant="ghost" size="sm" onClick={handleUseExample}>
                {msg("settings.keys.json_use_example")}
              </Button>
              <Button variant="ghost" size="sm" onClick={handleFormat} disabled={!text.trim()}>
                {msg("settings.keys.json_format")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowResults(true)}
                disabled={!text.trim()}
              >
                {msg("settings.keys.json_validate")}
              </Button>
            </div>
          </div>

          {showResults && result && (
            <div className="flex flex-col gap-2.5 animate-in fade-in-0">
              {result.errors.length > 0 ? (
                <div className="flex flex-col gap-1.5 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2.5">
                  <span className="text-xs font-medium text-destructive">
                    {formatMsg("settings.keys.json_errors_heading", {
                      count: result.errors.length,
                    })}
                  </span>
                  <ul className="flex flex-col gap-1">
                    {result.errors.map((err, i) => (
                      <li
                        key={i}
                        className="flex items-start gap-1.5 text-[0.6875rem] text-destructive/90"
                        dir="auto"
                      >
                        <AlertTriangle className="mt-px size-3 shrink-0" />
                        <span>{err}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-1.5 text-xs font-medium text-[#3f7d4f] dark:text-emerald-400">
                    <CheckCircle2 className="size-3.5 shrink-0" />
                    <span>
                      {formatMsg("settings.keys.json_valid", { count: result.connections.length })}
                    </span>
                  </div>

                  {result.warnings.length > 0 && (
                    <ul className="flex flex-col gap-1">
                      {result.warnings.map((warn, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-1.5 text-[0.6875rem] text-amber-700 dark:text-amber-400"
                          dir="auto"
                        >
                          <AlertTriangle className="mt-px size-3 shrink-0" />
                          <span>{warn}</span>
                        </li>
                      ))}
                    </ul>
                  )}

                  <div className="overflow-x-auto rounded-md border border-border/50">
                    <div className="grid min-w-[480px] grid-cols-[1.4fr_1.3fr_1.5fr_0.9fr] items-center gap-2 border-b border-border/50 bg-muted/40 px-3 py-1.5 text-[0.625rem] font-medium uppercase tracking-wide text-muted-foreground">
                      <span className="text-start">{msg("settings.keys.json_col_provider")}</span>
                      <span className="text-start">{msg("settings.keys.json_col_label")}</span>
                      <span className="text-start">{msg("settings.keys.json_col_base")}</span>
                      <span className="text-start">{msg("settings.keys.json_col_key")}</span>
                    </div>
                    {result.connections.map((c, i) => (
                      <div
                        key={i}
                        className="grid min-w-[480px] grid-cols-[1.4fr_1.3fr_1.5fr_0.9fr] items-center gap-2 px-3 py-2 text-xs [&:not(:last-child)]:border-b [&:not(:last-child)]:border-border/40"
                      >
                        <span className="flex min-w-0 items-center gap-1.5">
                          <ProviderLogo slug={c.provider} size={18} />
                          <span className="truncate text-foreground">
                            {providerLabel(c.provider)}
                          </span>
                        </span>
                        <span className="truncate text-muted-foreground" dir="auto">
                          {c.label || "—"}
                        </span>
                        <span
                          className="truncate font-mono text-[0.6875rem] text-muted-foreground"
                          dir="ltr"
                        >
                          {c.api_base || msg("settings.keys.json_base_default")}
                        </span>
                        <code
                          className="truncate font-mono text-[0.6875rem] text-muted-foreground"
                          dir="ltr"
                        >
                          {maskKey(c.api_key)}
                        </code>
                      </div>
                    ))}
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full"
                    onClick={handleImport}
                    disabled={busy}
                  >
                    {busy ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      formatMsg("settings.keys.json_import_count", {
                        count: result.connections.length,
                      })
                    )}
                  </Button>
                </>
              )}
            </div>
          )}

          <p className="flex items-start gap-1.5 text-[0.625rem] leading-relaxed text-muted-foreground/70">
            <ShieldCheck className="mt-px size-3 shrink-0" aria-hidden="true" />
            <span>{msg("settings.keys.json_security")}</span>
          </p>
        </div>
      )}
    </div>
  );
}
