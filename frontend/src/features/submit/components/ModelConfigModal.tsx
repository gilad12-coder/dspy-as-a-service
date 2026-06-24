"use client";

import * as React from "react";
import { Boxes, ChevronDown, X } from "lucide-react";
import { Dialog, DialogContent, DialogFooter } from "@/shared/ui/primitives/dialog";
import { DialogTitleRow } from "@/shared/ui/dialog-title-row";
import { Button } from "@/shared/ui/primitives/button";
import { Label } from "@/shared/ui/primitives/label";
import { Input } from "@/shared/ui/primitives/input";
import { Switch } from "@/shared/ui/primitives/switch";
import { Separator } from "@/shared/ui/primitives/separator";
import { ModelPicker, modelSupportsThinking } from "./ModelPicker";
import { NumberInput } from "@/shared/ui/number-input";
import { cn } from "@/shared/lib/utils";
import type { ModelConfig, CatalogModel } from "@/shared/types/api";
import { HelpTip } from "@/shared/ui/help-tip";
import { tip } from "@/shared/lib/tooltips";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";

interface ModelConfigModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  config: ModelConfig;
  onSave: (config: ModelConfig) => void;
  /** Label shown in the dialog header, e.g. the primary-model or reflection-model term. */
  roleLabel?: string;
  /** Catalog models for thinking detection */
  catalogModels?: CatalogModel[];
  /** Recently used configs — shown as quick-select at top */
  recentConfigs?: ModelConfig[];
  /** Remove a single recent config by its model name (rendered as a per-row X). */
  onRemoveRecent?: (name: string) => void;
  /**
   * When provided, renders a pinned "all available models" sentinel row at the
   * top of the dialog body. Clicking it closes the modal and invokes the
   * callback — the caller is expected to flip the matching grid-search flag.
   */
  onSelectAllAvailable?: () => void;
}

export function ModelConfigModal({
  open,
  onOpenChange,
  config,
  onSave,
  roleLabel = msg("auto.features.submit.components.modelconfigmodal.literal.1"),
  catalogModels,
  recentConfigs,
  onRemoveRecent,
  onSelectAllAvailable,
}: ModelConfigModalProps) {
  const [draft, setDraft] = React.useState<ModelConfig>(config);
  // Custom-connection section is collapsed for the common (built-in provider)
  // case; it auto-expands when the opened config already carries an endpoint
  // or key, so a populated field is never hidden behind a closed disclosure.
  const [connectionOpen, setConnectionOpen] = React.useState(false);

  // Sync draft when config changes externally (e.g. opening with different model)
  React.useEffect(() => {
    if (open) {
      setDraft(config);
      setConnectionOpen(!!(config.base_url || config.extra?.api_key));
    }
  }, [open, config]);

  const canThink = modelSupportsThinking(draft.name, catalogModels);
  const thinkingEnabled = !!draft.extra?.reasoning_effort;
  const reasoningEffort = (draft.extra?.reasoning_effort as string) ?? "medium";

  // Plaintext http to a remote host sends the API key unencrypted; localhost
  // loopback (Ollama, LM Studio, llama.cpp) is the common local-model case and
  // doesn't warrant the warning.
  const insecureRemoteEndpoint =
    /^http:\/\//i.test(draft.base_url ?? "") &&
    !/^http:\/\/(localhost|127\.0\.0\.1|\[?::1\]?)(?::|\/|$)/i.test(draft.base_url ?? "");

  const setThinking = (on: boolean) => {
    setDraft((p) => ({
      ...p,
      extra: on
        ? { ...p.extra, reasoning_effort: "medium" }
        : (() => {
            const rest = { ...p.extra };
            delete rest.reasoning_effort;
            return Object.keys(rest).length ? rest : undefined;
          })(),
    }));
  };

  const setEffort = (level: string) => {
    setDraft((p) => ({ ...p, extra: { ...p.extra, reasoning_effort: level } }));
  };

  const handleSave = () => {
    onSave(draft);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[85vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-2xl">
        <DialogTitleRow title={roleLabel} className="px-6 pt-6" />

        <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4">
          {onSelectAllAvailable &&
            (() => {
              const availableCount = catalogModels?.length ?? 0;
              const disabled = availableCount === 0;
              return (
                <>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => {
                      onSelectAllAvailable();
                      onOpenChange(false);
                    }}
                    className={cn(
                      "group flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-start transition-all",
                      disabled
                        ? "cursor-not-allowed border-border/40 bg-muted/20 opacity-60"
                        : "cursor-pointer border-primary/30 bg-primary/5 hover:border-primary/50 hover:bg-primary/10",
                    )}
                  >
                    <span
                      className={cn(
                        "flex size-9 shrink-0 items-center justify-center rounded-md",
                        disabled
                          ? "bg-muted text-muted-foreground"
                          : "bg-primary/10 text-primary group-hover:bg-primary/15",
                      )}
                    >
                      <Boxes className="size-4" />
                    </span>
                    <span className="flex min-w-0 flex-1 flex-col">
                      <span className="text-sm font-medium text-foreground">
                        {msg("auto.features.submit.components.modelconfigmodal.1")}
                      </span>
                      <span className="text-[0.6875rem] text-muted-foreground">
                        {disabled
                          ? msg("auto.features.submit.components.modelconfigmodal.literal.2")
                          : formatMsg(
                              "auto.features.submit.components.modelconfigmodal.template.1",
                              { p1: TERMS.optimization, p2: TERMS.model, p3: availableCount },
                            )}
                      </span>
                    </span>
                  </button>
                  <Separator />
                </>
              );
            })()}

          {recentConfigs && recentConfigs.length > 0 && (
            <div className="space-y-1.5">
              <Label className="text-[0.625rem] uppercase tracking-wide text-muted-foreground">
                {msg("auto.features.submit.components.modelconfigmodal.2")}
              </Label>
              <div className="flex gap-1.5 overflow-x-auto pb-1.5 scrollbar-thin" dir="ltr">
                {recentConfigs.map((rc, i) => {
                  const isActive = draft.name === rc.name;
                  return (
                    <div
                      key={`${rc.name}-${i}`}
                      className={cn(
                        "group/recent flex shrink-0 items-center gap-1.5 rounded-md border ps-2 pe-1 py-1 text-[0.6875rem] font-mono transition-all",
                        isActive
                          ? "border-primary/50 bg-primary/5 text-foreground"
                          : "border-border/40 bg-muted/30 text-muted-foreground hover:border-primary/40 hover:text-foreground hover:bg-muted/50",
                      )}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          setDraft({ ...rc });
                          // Reveal the connection fields when the restored config
                          // carries a custom endpoint, so the base URL isn't
                          // hidden behind the collapsed disclosure.
                          setConnectionOpen(!!(rc.base_url || rc.extra?.api_key));
                        }}
                        className="flex items-center gap-1.5 cursor-pointer outline-none"
                      >
                        <span className="truncate max-w-[120px]">{rc.name.split("/").pop()}</span>
                        <span className="text-[9px] opacity-60">{rc.temperature?.toFixed(1)}</span>
                      </button>
                      {onRemoveRecent && (
                        <button
                          type="button"
                          aria-label={formatMsg(
                            "auto.features.submit.components.modelconfigmodal.recent.remove",
                            { model: rc.name.split("/").pop() ?? rc.name },
                          )}
                          onClick={(e) => {
                            e.stopPropagation();
                            onRemoveRecent(rc.name);
                          }}
                          className="ms-0.5 inline-flex h-4 w-4 items-center justify-center rounded text-muted-foreground/60 hover:text-destructive hover:bg-destructive/10 transition-colors cursor-pointer"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
              <Separator />
            </div>
          )}

          <button
            type="button"
            onClick={() => setConnectionOpen((o) => !o)}
            aria-expanded={connectionOpen}
            className="flex w-full cursor-pointer items-center justify-between gap-1.5 text-[0.625rem] uppercase tracking-wide text-muted-foreground transition-colors hover:text-foreground"
          >
            <HelpTip text={tip("model_config.connection_section")} className="cursor-pointer">
              {msg("auto.features.submit.components.modelconfigmodal.section.connection")}
            </HelpTip>
            <ChevronDown
              className={cn(
                "size-3 shrink-0 transition-transform duration-150",
                connectionOpen && "rotate-180",
              )}
            />
          </button>

          {connectionOpen && (
            <div className="space-y-4 animate-in fade-in-0 slide-in-from-top-1">
              <div className="space-y-2">
                <Label htmlFor="modelConfigBaseUrl">
                  <HelpTip text={tip("model_config.base_url")}>
                    {msg("auto.features.submit.components.steps.modelstep.8")}
                  </HelpTip>
                </Label>
                <Input
                  id="modelConfigBaseUrl"
                  dir="ltr"
                  type="url"
                  inputMode="url"
                  autoComplete="off"
                  placeholder="https://my-host:1234/v1"
                  value={draft.base_url ?? ""}
                  onChange={(e) => {
                    const next = e.target.value.trim();
                    setDraft((p) => ({ ...p, base_url: next ? next : undefined }));
                  }}
                />
                {insecureRemoteEndpoint && (
                  <p className="text-[0.625rem] text-amber-700 dark:text-amber-400">
                    {msg("auto.features.submit.components.steps.modelstep.12")}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="modelConfigApiKey">
                  <HelpTip text={tip("model_config.api_key")}>
                    {msg("auto.features.submit.components.steps.modelstep.10")}
                  </HelpTip>
                </Label>
                <Input
                  id="modelConfigApiKey"
                  dir="ltr"
                  type="password"
                  placeholder="sk-..."
                  autoComplete="new-password"
                  value={(draft.extra?.api_key as string | undefined) ?? ""}
                  onChange={(e) => {
                    const next = e.target.value;
                    setDraft((p) => {
                      const rest = { ...p.extra };
                      if (next) rest.api_key = next;
                      else delete rest.api_key;
                      return { ...p, extra: Object.keys(rest).length ? rest : undefined };
                    });
                  }}
                />
              </div>
            </div>
          )}

          <Separator />

          <div className="space-y-2">
            <Label>
              <HelpTip text={tip("model_config.model")}>
                {msg("auto.features.submit.components.modelconfigmodal.4")}
              </HelpTip>
            </Label>
            <ModelPicker
              value={draft.name}
              onChange={(next) => {
                setDraft((p) => ({ ...p, name: next }));
                if (!modelSupportsThinking(next, catalogModels)) {
                  setDraft((p) => {
                    const rest = { ...p.extra };
                    delete rest.reasoning_effort;
                    return { ...p, extra: Object.keys(rest).length ? rest : undefined };
                  });
                }
              }}
              discoverUrl={draft.base_url?.trim() || undefined}
              discoverApiKey={(draft.extra?.api_key as string | undefined) || undefined}
              placeholder={msg("auto.features.submit.components.modelconfigmodal.literal.3")}
            />
          </div>

          <Separator />

          <Label className="text-[0.625rem] uppercase tracking-wide text-muted-foreground">
            {msg("auto.features.submit.components.modelconfigmodal.section.parameters")}
          </Label>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>
                <HelpTip text={tip("model_config.temperature")}>
                  {msg("auto.features.submit.components.modelconfigmodal.5")}
                </HelpTip>
              </Label>
              <span className="text-xs font-mono text-muted-foreground">
                {draft.temperature?.toFixed(1) ?? "0.7"}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={draft.temperature ?? 0.7}
              onChange={(e) => setDraft((p) => ({ ...p, temperature: parseFloat(e.target.value) }))}
              className="w-full h-2 bg-muted rounded-full appearance-none cursor-pointer accent-primary"
              dir="auto"
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>
                <HelpTip text={tip("model_config.top_p")}>
                  {msg("auto.features.submit.components.modelconfigmodal.6")}
                </HelpTip>
              </Label>
              <span className="text-xs font-mono text-muted-foreground">
                {draft.top_p?.toFixed(2) ?? "—"}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={draft.top_p ?? 1}
              onChange={(e) => setDraft((p) => ({ ...p, top_p: parseFloat(e.target.value) }))}
              className="w-full h-2 bg-muted rounded-full appearance-none cursor-pointer accent-primary"
              dir="auto"
            />
          </div>

          <div className="space-y-2">
            <Label>
              <HelpTip text={tip("model_config.max_tokens")}>
                {msg("auto.features.submit.components.modelconfigmodal.7")}
              </HelpTip>
            </Label>
            <NumberInput
              min={1}
              step={256}
              value={draft.max_tokens ?? ""}
              onChange={(v) => setDraft((p) => ({ ...p, max_tokens: v }))}
            />
          </div>

          {canThink && (
            <>
              <Separator />
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>{msg("auto.features.submit.components.modelconfigmodal.8")}</Label>
                  <Switch checked={thinkingEnabled} onCheckedChange={setThinking} />
                </div>
                {thinkingEnabled && (
                  <div className="space-y-2 p-3 border rounded-lg bg-muted/30">
                    <Label>{msg("auto.features.submit.components.modelconfigmodal.9")}</Label>
                    <div className="flex rounded-lg bg-muted p-0.5 w-full">
                      {(
                        [
                          [
                            "low",
                            msg("auto.features.submit.components.modelconfigmodal.literal.4"),
                          ],
                          [
                            "medium",
                            msg("auto.features.submit.components.modelconfigmodal.literal.5"),
                          ],
                          [
                            "high",
                            msg("auto.features.submit.components.modelconfigmodal.literal.6"),
                          ],
                        ] as const
                      ).map(([val, label]) => (
                        <button
                          key={val}
                          type="button"
                          onClick={() => setEffort(val)}
                          className={cn(
                            "flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors text-center cursor-pointer",
                            reasoningEffort === val
                              ? "bg-background text-foreground shadow-sm"
                              : "text-muted-foreground hover:text-foreground",
                          )}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        <DialogFooter className="border-t border-border/40 px-6 pb-6 pt-4">
          <Button variant="outline" onClick={() => onOpenChange(false)} className="flex-1">
            {msg("auto.features.submit.components.modelconfigmodal.10")}
          </Button>
          <Button onClick={handleSave} disabled={!draft.name.trim()} className="flex-1">
            {msg("auto.features.submit.components.modelconfigmodal.11")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
