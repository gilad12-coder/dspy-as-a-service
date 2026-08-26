import { useEffect, useState } from "react";
import { getQueueStatus } from "@/shared/lib/api";
import type { QueueStatusResponse } from "@/shared/types/api";

const POLL_INTERVAL_MS = 30_000;

export function useQueueStatus(): QueueStatusResponse | null {
  const [queueStatus, setQueueStatus] = useState<QueueStatusResponse | null>(null);

  useEffect(() => {
    const load = () => {
      getQueueStatus()
        .then(setQueueStatus)
        .catch(() => {});
    };
    load();
    // Pause polling in background tabs and refresh immediately on return
    // (mirrors the visibility-gated ticking in LiveElapsed).
    const tick = () => {
      if (document.visibilityState === "visible") load();
    };
    const interval = setInterval(tick, POLL_INTERVAL_MS);
    const onVisibility = () => {
      if (document.visibilityState === "visible") load();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return queueStatus;
}
