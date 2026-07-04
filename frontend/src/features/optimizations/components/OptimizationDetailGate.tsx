"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { XCircle } from "lucide-react";

import {
  getJob,
  getPublicOptimization,
  setApiAuthToken,
  type SharedOptimizationData,
} from "@/shared/lib/api";
import { formatMsg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
// Leaf import on purpose — the tutorial barrel deliberately does not re-export
// the demo fixtures (see features/tutorial/index.ts).
// eslint-disable-next-line no-restricted-imports -- deliberate leaf import; see above
import { DEMO_OPTIMIZATION_ID, DEMO_GRID_OPTIMIZATION_ID } from "@/features/tutorial/lib/demo-data";
import { OptimizationDetailView } from "./OptimizationDetailView";
import { OptimizationDetailSkeleton } from "./OptimizationDetailSkeleton";

type GateState =
  | { mode: "loading" }
  | { mode: "owned" }
  | { mode: "public"; data: SharedOptimizationData }
  | { mode: "notfound" };

/**
 * Decides which detail view to render for ``/optimizations/[id]``: the full
 * owner/member view when the caller can access the run, or a scrubbed read-only
 * public view when they can't but the run is in the public Explore corpus.
 *
 * Explore lists every ``is_private=false`` run, so a non-owner clicking one used
 * to 404 on the access-gated detail route. This probes ``getJob`` first (with the
 * bearer attached, so the owner path is unchanged) and only on a no-access
 * failure falls back to the public composite — keeping public discoverability
 * and view access in sync. Demo ids skip the probe (they never hit the network).
 */
export function OptimizationDetailGate() {
  const { id } = useParams<{ id: string }>();
  const { data: session, status } = useSession();
  const [state, setState] = useState<GateState>({ mode: "loading" });

  const isDemo = id === DEMO_OPTIMIZATION_ID || id === DEMO_GRID_OPTIMIZATION_ID;

  useEffect(() => {
    if (isDemo) {
      setState({ mode: "owned" });
      return;
    }
    if (status === "loading") return;
    let cancelled = false;
    setState({ mode: "loading" });
    // Attach the bearer before probing — effects run child-before-parent, so the
    // root ApiAuthTokenBridge may not have synced it yet; without this the
    // owner's getJob could 401 and wrongly fall through to the public view.
    if (session?.backendAccessToken) setApiAuthToken(session.backendAccessToken);
    getJob(id)
      .then(() => {
        if (!cancelled) setState({ mode: "owned" });
      })
      .catch(() => {
        // No access (404) — try the public corpus before giving up.
        getPublicOptimization(id)
          .then((data) => {
            if (!cancelled) setState({ mode: "public", data });
          })
          .catch(() => {
            if (!cancelled) setState({ mode: "notfound" });
          });
      });
    return () => {
      cancelled = true;
    };
  }, [id, isDemo, status, session?.backendAccessToken]);

  if (state.mode === "loading") return <OptimizationDetailSkeleton />;
  if (state.mode === "public") return <OptimizationDetailView shareData={state.data} />;
  if (state.mode === "notfound") {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <XCircle className="size-12 text-destructive" />
        <p className="text-lg text-muted-foreground">
          {formatMsg("auto.app.optimizations.id.page.template.2", { p1: TERMS.optimization })}
        </p>
      </div>
    );
  }
  return <OptimizationDetailView />;
}
