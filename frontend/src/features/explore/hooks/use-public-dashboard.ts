"use client";

import { useEffect, useRef, useState } from "react";
import {
  getPublicDashboard,
  invalidateCache,
  type PublicDashboardPoint,
} from "@/shared/lib/api";
import { POLL_CATCHUP_EPSILON_MS, POLL_INTERVAL_MS } from "../constants";

export interface PublicDashboardState {
  points: PublicDashboardPoint[];
  loading: boolean;
  error: string | null;
}

export function usePublicDashboard(): PublicDashboardState {
  const [state, setState] = useState<PublicDashboardState>({
    points: [],
    loading: true,
    error: null,
  });
  const lastTickRef = useRef<number>(0);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      if (cancelled) return;
      try {
        const data = await getPublicDashboard();
        if (cancelled) return;
        lastTickRef.current = Date.now();
        setState({ points: data.points, loading: false, error: null });
      } catch (err) {
        if (cancelled) return;
        setState((current) => ({
          ...current,
          loading: false,
          error: err instanceof Error ? err.message : "load failed",
        }));
      }
    };

    const tick = () => {
      if (cancelled) return;
      if (typeof document !== "undefined" && document.visibilityState !== "visible") return;
      invalidateCache("/dashboard/public");
      void load();
    };

    const onVisibility = () => {
      if (typeof document === "undefined" || document.visibilityState !== "visible") return;
      const elapsed = Date.now() - lastTickRef.current;
      if (elapsed >= POLL_INTERVAL_MS - POLL_CATCHUP_EPSILON_MS) tick();
    };

    void load();
    const timer = setInterval(tick, POLL_INTERVAL_MS);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelled = true;
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return state;
}
