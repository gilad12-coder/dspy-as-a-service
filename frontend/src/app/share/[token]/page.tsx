"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { CircleNotch } from "@/shared/ui/icons";

import { claimSharedOptimization, setApiAuthToken } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { Button } from "@/shared/ui/primitives/button";

/**
 * Share-link redeemer (Google-Drive semantics). The route is login-gated, so the
 * recipient is authenticated by the time this mounts: it attaches the bearer,
 * redeems the token (durably granting the link's viewer/editor tier on the
 * caller's account), then replaces into the normal `/optimizations/{id}` page —
 * the same surface the owner uses, gated to the granted role. There is no
 * separate "shared" view: once redeemed, the run simply lives in the recipient's
 * account (and the unified table / "משותף איתי" sidebar).
 */
export default function SharePage() {
  const { token } = useParams<{ token: string }>();
  const router = useRouter();
  const { data: session, status } = useSession();
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    // Wait for the session to resolve, then attach the bearer before redeeming.
    // Effects run child-before-parent, so the root ApiAuthTokenBridge may not
    // have synced the token on this tick — setting it here keeps the claim from
    // going out anonymously (which the backend would reject behind the login
    // gate / floor to no-access).
    if (status === "loading") return;
    let cancelled = false;
    if (session?.backendAccessToken) setApiAuthToken(session.backendAccessToken);
    claimSharedOptimization(token)
      .then((res) => {
        if (!cancelled) router.replace(`/optimizations/${res.optimization_id}`);
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
      <div className="flex flex-col items-center justify-center min-h-screen gap-3 px-6 text-center">
        <h1 className="text-lg font-semibold">{msg("share.not_found_title")}</h1>
        <p className="text-sm text-muted-foreground">{msg("share.not_found_body")}</p>
        <Button asChild variant="outline" className="mt-2">
          <Link href="/">{msg("not_found.back_dashboard")}</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen">
      <CircleNotch className="size-8 animate-spin text-primary" />
    </div>
  );
}
