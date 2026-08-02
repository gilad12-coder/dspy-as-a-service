"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { CircleNotch, XCircle } from "@/shared/ui/icons";

import {
  getTaggerSession,
  takeTaggerSession,
  setApiAuthToken,
  type TaggerSessionDetail,
} from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { PageContainer } from "@/shared/layout/page-container";
import { TaggerBackLink } from "./TaggerBackLink";
import { TaggerView } from "./TaggerView";

type GateState =
  | { mode: "loading" }
  | { mode: "ready"; session: TaggerSessionDetail }
  | { mode: "notfound" };

/**
 * Resolves ``/tagger/[id]``: fetches the caller's saved session and hands its
 * full state to {@link TaggerView} to resume annotating, or shows a not-found
 * state when the id is unknown or owned by someone else. Mirrors
 * ``OptimizationDetailGate`` — the bearer is attached before the probe because
 * effects run child-before-parent and the root token bridge may not have synced.
 */
export function TaggerSessionGate() {
  const { id } = useParams<{ id: string }>();
  const { data: session, status } = useSession();
  const [state, setState] = useState<GateState>({ mode: "loading" });

  useEffect(() => {
    if (status === "loading") return;
    // Resume instantly from the wizard's same-tab handoff when present; a genuine
    // reload or a return from elsewhere finds none and fetches from the server.
    const handed = takeTaggerSession(id);
    if (handed) {
      setState({ mode: "ready", session: handed });
      return;
    }
    let cancelled = false;
    setState({ mode: "loading" });
    if (session?.backendAccessToken) setApiAuthToken(session.backendAccessToken);
    getTaggerSession(id)
      .then((detail) => {
        if (!cancelled) setState({ mode: "ready", session: detail });
      })
      .catch(() => {
        if (!cancelled) setState({ mode: "notfound" });
      });
    return () => {
      cancelled = true;
    };
  }, [id, status, session?.backendAccessToken]);

  if (state.mode === "loading") {
    return (
      <PageContainer full>
        <div className="flex items-center justify-center min-h-[60vh]">
          <CircleNotch className="size-8 animate-spin text-primary" />
        </div>
      </PageContainer>
    );
  }
  if (state.mode === "notfound") {
    return (
      <PageContainer full>
        <div className="mb-3">
          <TaggerBackLink />
        </div>
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <XCircle className="size-12 text-destructive" />
          <p className="text-lg text-muted-foreground">{msg("tagger.session.notfound")}</p>
        </div>
      </PageContainer>
    );
  }
  // Remount on id change so the hook re-seeds from the new session's state.
  // TaggerView renders its own back link, so none is added here.
  return <TaggerView key={state.session.id} initialSession={state.session} />;
}
