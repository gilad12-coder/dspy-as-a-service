import { DataHubTabs } from "@/shared/ui/data-hub-tabs";
import { ListPageSkeleton } from "@/shared/ui/list-page-skeleton";

export default function Loading() {
  return (
    <div className="pb-16">
      <DataHubTabs active="datasets" />
      <ListPageSkeleton />
    </div>
  );
}
