import * as React from "react";

interface SettingsRowProps {
  label: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}

/**
 * A label/description-on-the-leading-side, control-on-the-trailing-side row.
 *
 * The shared building block for the Settings modal and any modal that wants the
 * same visual rhythm (e.g. the share dialog) — a bottom border per row, a small
 * leading icon, and a right-aligned control slot.
 */
export function SettingsRow({ label, description, icon: Icon, children }: SettingsRowProps) {
  return (
    <div className="flex flex-col items-stretch gap-3 border-b border-border/40 py-3 last:border-b-0 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
      <div className="flex items-start gap-3 flex-1 min-w-0">
        {Icon && (
          <Icon className="size-4 mt-0.5 text-muted-foreground shrink-0" aria-hidden="true" />
        )}
        <div className="flex flex-col gap-0.5 min-w-0">
          <span className="text-sm font-medium text-foreground">{label}</span>
          {description && <span className="text-xs text-muted-foreground/80">{description}</span>}
        </div>
      </div>
      <div className="flex min-w-0 w-full flex-wrap items-center gap-2 [&>*]:max-w-full sm:w-auto sm:shrink-0 sm:justify-end">
        {children}
      </div>
    </div>
  );
}
