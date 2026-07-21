"use client";

import {
  SessionProvider as NextAuthSessionProvider,
  getSession,
  useSession,
} from "next-auth/react";
import type { ComponentProps } from "react";
import * as React from "react";
import { setApiAuthToken, setApiAuthTokenRefresher } from "@/shared/lib/api";

type SessionProviderProps = ComponentProps<typeof NextAuthSessionProvider>;

export function SessionProvider(props: SessionProviderProps) {
  // Refetch every 5 min to keep ``backendAccessToken`` fresh; the backend
  // bearer JWT has a 15-min TTL (BACKEND_AUTH_TOKEN_TTL_SECONDS=900) and
  // without polling the in-memory token expires while the modal sits open.
  return (
    <NextAuthSessionProvider
      refetchInterval={300}
      refetchOnWindowFocus={false}
      refetchWhenOffline={false}
      {...props}
    >
      <ApiAuthTokenBridge />
      {props.children}
    </NextAuthSessionProvider>
  );
}

function ApiAuthTokenBridge() {
  const { data: session } = useSession();

  React.useEffect(() => {
    setApiAuthToken(session?.backendAccessToken);
  }, [session?.backendAccessToken]);

  // On a 401 the API layer calls this to mint a fresh bearer by re-fetching
  // the session — recovers a tab whose cached token expired while idle
  // during a long run, without forcing a full reload.
  React.useEffect(() => {
    if (!session?.backendAccessToken) {
      setApiAuthTokenRefresher(undefined);
      return;
    }
    setApiAuthTokenRefresher(async () => {
      const fresh = await getSession();
      return fresh?.backendAccessToken;
    });
    return () => setApiAuthTokenRefresher(undefined);
  }, [session?.backendAccessToken]);

  return null;
}
