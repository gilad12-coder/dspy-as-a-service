"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { CircleNotch } from "@/shared/ui/icons";

import { claimSharedTaggerSession, setApiAuthToken } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { Button } from "@/shared/ui/primitives/button";

/**
 * Labeling-session share-link redeemer (Google-Drive semantics). The route is
 * login-gated, so the recipient is authenticated by the time this mounts: it
 * attaches the bearer, redeems the token (durably granting the link's
 * viewer/editor tier on the caller's account), then replaces into
 * ``/tagger/<id>`` — the shared session itself. Once redeemed the session also
 * lists in the recipient's labeling-sessions chooser.
 */
export default function TaggerSharePage() {
  const { token } = useParams<{ token: string }>();
  const router = useRouter();
  const { data: session, status } = useSession();
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    // Wait for the session to resolve, then attach the bearer before redeeming —
    // effects run child-before-parent so the root bridge may not have synced the
    // token yet; setting it here keeps the claim from going out anonymously.
    if (status === "loading") return;
    let cancelled = false;
    if (session?.backendAccessToken) setApiAuthToken(session.backendAccessToken);
    claimSharedTaggerSession(token)
      .then((res) => {
        if (!cancelled) router.replace(`/tagger/${res.session_id}`);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [token, status, session?.backendAccessToken, router]);

  if (failed) {
    return (
      <div className="flex min-h-dvh flex-col items-center justify-center gap-3 px-6 text-center">
        <h1 className="text-lg font-semibold">{msg("datasets.share.not_found_title")}</h1>
        <p className="text-sm text-muted-foreground">{msg("datasets.share.not_found_body")}</p>
        <Button asChild variant="outline" className="mt-2 min-h-[44px]">
          <Link href="/">{msg("not_found.back_dashboard")}</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="flex min-h-dvh items-center justify-center">
      <CircleNotch className="size-8 animate-spin text-primary" />
    </div>
  );
}
