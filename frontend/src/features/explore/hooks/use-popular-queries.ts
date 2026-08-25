"use client";

import * as React from "react";
import { getPopularQueries, type PopularQuery } from "@/shared/lib/api";

/** Fetch deployment-local popular public-corpus searches. */
export function usePopularQueries(): PopularQuery[] {
  const [queries, setQueries] = React.useState<PopularQuery[]>([]);

  React.useEffect(() => {
    let cancelled = false;
    void getPopularQueries()
      .then((data) => {
        if (!cancelled) setQueries(data.queries);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  return queries;
}
