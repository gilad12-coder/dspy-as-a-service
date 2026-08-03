"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { CaretRight, Coins, Database, Tag } from "@/shared/ui/icons";
import { Card } from "@/shared/ui/primitives/card";
import { formatCredits, useCredits } from "@/features/billing";
import { useSettingsModal } from "@/features/settings";
import { formatBytes } from "@/shared/lib/formatters";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";
import { cn } from "@/shared/lib/utils";
import { useWorkspaceSummary } from "../hooks/use-workspace-summary";

/** Chrome shared by the three workspace cards: icon tile, title, count, body. */
function WorkspaceCard({
  icon,
  title,
  count,
  href,
  onOpen,
  children,
}: {
  icon: ReactNode;
  title: string;
  count?: number;
  href?: string;
  onOpen?: () => void;
  children: ReactNode;
}) {
  const header = (
    <span className="flex min-w-0 items-center gap-2">
      <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-accent text-foreground/70">
        {icon}
      </span>
      <span className="truncate text-xs font-semibold text-foreground">{title}</span>
      {count !== undefined && (
        <span className="text-xs font-medium text-muted-foreground tabular-nums">{count}</span>
      )}
      <CaretRight
        aria-hidden="true"
        className="ms-auto size-3.5 shrink-0 text-muted-foreground/40 transition-[color,transform] duration-150 group-hover/ws:translate-x-0.5 group-hover/ws:text-primary rtl:rotate-180 rtl:group-hover/ws:-translate-x-0.5"
      />
    </span>
  );
  return (
    <Card className="group/ws min-w-0 gap-0 p-4">
      {href ? (
        <Link
          href={href}
          className="rounded-md focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
        >
          {header}
        </Link>
      ) : (
        <button
          type="button"
          onClick={onOpen}
          className="w-full cursor-pointer rounded-md text-start focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
        >
          {header}
        </button>
      )}
      <div className="mt-3 flex flex-col gap-1.5">{children}</div>
    </Card>
  );
}

function ItemRow({ href, name, meta }: { href: string; name: string; meta: string }) {
  return (
    <Link
      href={href}
      className="flex items-baseline justify-between gap-3 rounded-md text-xs hover:text-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
    >
      <span className="min-w-0 truncate text-foreground/80 hover:underline underline-offset-2" dir="auto">
        {name}
      </span>
      <span className="shrink-0 text-muted-foreground tabular-nums" dir="ltr">
        {meta}
      </span>
    </Link>
  );
}

function EmptyHint({ text }: { text: string }) {
  return <p className="text-xs text-muted-foreground">{text}</p>;
}

/**
 * Compact cards surfacing the workspace surfaces the run-centric dashboard
 * predates: labeling sessions, the dataset library (with its storage meter),
 * and the credit wallet. Every card is one cheap, mostly-cached call — the
 * strip renders nothing for a section whose fetch failed rather than
 * blocking the page.
 */
export function WorkspaceStrip() {
  const { tagging, datasets, loading } = useWorkspaceSummary();
  const { wallet, loading: walletLoading } = useCredits();
  const { openTo } = useSettingsModal();
  const locale = getActiveIntlLocale();

  if (loading) {
    return (
      <div className="grid gap-3 sm:gap-4 md:grid-cols-3" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-28 animate-pulse rounded-2xl border border-border/40 bg-card/60" />
        ))}
      </div>
    );
  }
  if (!tagging && !datasets && walletLoading) return null;

  const usagePct = datasets
    ? Math.min(100, Math.round((datasets.usage.used_bytes / Math.max(1, datasets.usage.quota_bytes)) * 100))
    : 0;

  return (
    <div className={cn("grid gap-3 sm:gap-4", "md:grid-cols-2 lg:grid-cols-3")}>
      {tagging && (
        <WorkspaceCard
          icon={<Tag className="size-3.5" aria-hidden="true" />}
          title={msg("dashboard.workspace.tagging.title")}
          count={tagging.total}
          href="/tagger"
        >
          {tagging.recent.length === 0 ? (
            <EmptyHint text={msg("dashboard.workspace.tagging.cta")} />
          ) : (
            tagging.recent.map((s) => (
              <ItemRow
                key={s.id}
                href={`/tagger/${s.id}`}
                name={s.name}
                meta={`${s.tagged_count}/${s.row_count}`}
              />
            ))
          )}
        </WorkspaceCard>
      )}

      {datasets && (
        <WorkspaceCard
          icon={<Database className="size-3.5" aria-hidden="true" />}
          title={msg("dashboard.workspace.datasets.title")}
          count={datasets.total}
          href="/datasets"
        >
          {datasets.recent.length === 0 ? (
            <EmptyHint text={msg("dashboard.workspace.datasets.cta")} />
          ) : (
            datasets.recent.map((d) => (
              <ItemRow
                key={d.id}
                href={`/datasets/${d.id}/edit?name=${encodeURIComponent(d.name)}`}
                name={d.name}
                meta={formatMsg("datasets.count.rows", { count: d.row_count })}
              />
            ))
          )}
          <div className="mt-1 flex items-center gap-2">
            <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full bg-primary/50" style={{ width: `${usagePct}%` }} />
            </div>
            <span className="shrink-0 text-[0.6875rem] text-muted-foreground tabular-nums" dir="ltr">
              {formatBytes(datasets.usage.used_bytes)} / {formatBytes(datasets.usage.quota_bytes)}
            </span>
          </div>
        </WorkspaceCard>
      )}

      {!walletLoading && (
        <WorkspaceCard
          icon={<Coins className="size-3.5" aria-hidden="true" />}
          title={msg("dashboard.workspace.credits.title")}
          onOpen={() => openTo("billing")}
        >
          <p className="text-xl font-bold leading-none tracking-tight text-foreground tabular-nums">
            {formatCredits(wallet.paidBalanceCredits + wallet.freeGrant.creditsRemaining, locale)}
          </p>
          <p className="text-xs text-muted-foreground">
            {formatMsg("dashboard.workspace.credits.free", {
              remaining: formatCredits(wallet.freeGrant.creditsRemaining, locale),
              total: formatCredits(wallet.freeGrant.creditsTotal, locale),
            })}
          </p>
        </WorkspaceCard>
      )}
    </div>
  );
}
