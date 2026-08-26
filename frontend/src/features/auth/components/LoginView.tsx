"use client";

import { useEffect, useState } from "react";
import { getProviders, signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, CircleNotch } from "@/shared/ui/icons";
import { AnimatedWordmark } from "@/shared/ui/animated-wordmark";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent } from "@/shared/ui/primitives/card";
import { Input } from "@/shared/ui/primitives/input";
import { Label } from "@/shared/ui/primitives/label";
import { msg } from "@/shared/lib/messages";
import { LoginHalo } from "./LoginHalo";

const ENTER_EASE = [0.16, 1, 0.3, 1] as const;

function postLoginTarget(): string {
  if (typeof window === "undefined") return "/";
  const callback = new URLSearchParams(window.location.search).get("callbackUrl");
  if (!callback) return "/";
  try {
    const url = new URL(callback, window.location.origin);
    if (url.origin === window.location.origin) return url.pathname + url.search + url.hash;
  } catch {
    return "/";
  }
  return "/";
}

function LoginHeader() {
  return (
    <div className="w-[min(90vw,520px)]">
      <AnimatedWordmark fluid autoMorph autoMorphDuration={10000} morphSpeed={250} />
    </div>
  );
}

type LoginMode = "loading" | "sso" | "sso-error" | "local";

export function LoginView() {
  const router = useRouter();
  const [mode, setMode] = useState<LoginMode>("loading");
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const localRequested = params.get("local") === "1";
    const authError = params.has("error");
    void getProviders()
      .then((providers) => {
        if (localRequested || !providers?.adfs) {
          setMode("local");
          return;
        }
        if (authError) {
          setMode("sso-error");
          return;
        }
        setMode("sso");
        void signIn("adfs", { callbackUrl: postLoginTarget() });
      })
      .catch(() => setMode("local"));
  }, []);

  const handleLocalLogin = async (event: React.FormEvent) => {
    event.preventDefault();
    const normalized = username.trim();
    if (!normalized) return;
    setError("");
    setLoading(true);
    const result = await signIn("local", {
      username: normalized,
      redirect: false,
    });
    setLoading(false);
    if (result?.error) {
      setError(msg("auth.login.error"));
      return;
    }
    router.push(postLoginTarget());
    router.refresh();
  };

  const showLocalFallback = () => {
    setError("");
    setMode("local");
  };

  return (
    <div className="relative flex min-h-dvh w-full items-center justify-center px-4 py-10">
      <LoginHalo />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-[1]"
        style={{
          background:
            "radial-gradient(58% 48% at 50% 44%, rgba(250,248,245,0.9) 0%, rgba(250,248,245,0.4) 46%, transparent 76%)",
        }}
      />

      <motion.div
        initial={{ opacity: 0, y: 18, scale: 0.985 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: ENTER_EASE }}
        className="relative z-10 w-full max-w-[420px]"
      >
        <div className="flex flex-col items-center">
          <LoginHeader />

          {(mode === "loading" || mode === "sso") && (
            <div className="mt-9 flex flex-col items-center gap-4 text-sm text-muted-foreground">
              <span className="flex items-center gap-2">
                <CircleNotch className="size-4 animate-spin" />
                {msg("auth.login.sso_loading")}
              </span>
            </div>
          )}

          {mode === "sso-error" && (
            <Card className="mt-9 w-full">
              <CardContent className="space-y-4 px-6 text-center">
                <p className="text-sm text-muted-foreground">{msg("auth.login.sso_failed")}</p>
                <Button
                  className="w-full"
                  onClick={() => void signIn("adfs", { callbackUrl: postLoginTarget() })}
                >
                  {msg("auth.login.sso_retry")}
                </Button>
                <button
                  type="button"
                  onClick={showLocalFallback}
                  className="inline-flex min-h-11 cursor-pointer items-center gap-1 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
                >
                  {msg("auth.login.local_fallback")}
                  <ArrowLeft className="size-4" aria-hidden="true" />
                </button>
              </CardContent>
            </Card>
          )}

          {mode === "local" && (
            <Card className="mt-9 w-full">
              <CardContent className="px-6">
                <form
                  onSubmit={handleLocalLogin}
                  className="space-y-4"
                  aria-label={msg("auth.login.form_aria")}
                >
                  <div className="space-y-2">
                    <Label htmlFor="login-username">{msg("auth.login.username")}</Label>
                    <Input
                      id="login-username"
                      value={username}
                      onChange={(event) => setUsername(event.target.value)}
                      placeholder={msg("auth.login.username_placeholder")}
                      autoFocus
                      autoComplete="username"
                      dir="auto"
                      className="h-11 placeholder:text-right"
                    />
                  </div>

                  <AnimatePresence initial={false}>
                    {error && (
                      <motion.p
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="text-center text-sm text-destructive"
                        role="alert"
                      >
                        {error}
                      </motion.p>
                    )}
                  </AnimatePresence>

                  <Button
                    type="submit"
                    size="lg"
                    disabled={loading || !username.trim()}
                    className="h-11 w-full gap-2"
                  >
                    {loading && <CircleNotch className="size-4 animate-spin" />}
                    {msg("auth.login.submit")}
                  </Button>
                </form>
              </CardContent>
            </Card>
          )}
        </div>
      </motion.div>
    </div>
  );
}
