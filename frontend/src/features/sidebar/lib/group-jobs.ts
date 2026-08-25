import type { SidebarJobItem } from "@/shared/lib/api";
import { isActiveStatus } from "@/shared/constants/job-status";
import { msg } from "@/shared/lib/messages";
import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";

export interface JobGroup {
  label: string;
  jobs: SidebarJobItem[];
}

/**
 * Bucket jobs into pinned → active → concrete calendar dates.
 * Empty buckets are dropped.
 */
export function groupJobsByRecency(jobs: SidebarJobItem[], now: Date = new Date()): JobGroup[] {
  void now;
  const groups: JobGroup[] = [];
  const pinned: SidebarJobItem[] = [];
  const active: SidebarJobItem[] = [];
  const dated = new Map<string, JobGroup>();

  for (const job of jobs) {
    if (job.pinned) {
      pinned.push(job);
      continue;
    }
    if (isActiveStatus(job.status)) {
      active.push(job);
      continue;
    }
    const created = new Date(job.created_at ?? 0);
    const validDate = !Number.isNaN(created.getTime());
    const key = validDate
      ? `${created.getFullYear()}-${String(created.getMonth() + 1).padStart(2, "0")}-${String(
          created.getDate(),
        ).padStart(2, "0")}`
      : "unknown";
    const label = validDate
      ? created.toLocaleDateString(getActiveIntlLocale(), {
          day: "2-digit",
          month: "2-digit",
          year: "numeric",
        })
      : msg("auto.features.sidebar.lib.group.jobs.literal.6");
    const group = dated.get(key) ?? { label, jobs: [] };
    group.jobs.push(job);
    dated.set(key, group);
  }

  if (pinned.length)
    groups.push({ label: msg("auto.features.sidebar.lib.group.jobs.literal.1"), jobs: pinned });
  if (active.length)
    groups.push({ label: msg("auto.features.sidebar.lib.group.jobs.literal.2"), jobs: active });
  groups.push(...dated.values());

  return groups;
}
