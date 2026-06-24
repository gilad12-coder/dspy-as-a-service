"use client";

import { useEffect, useState } from "react";
import { signIn, getProviders } from "next-auth/react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Github } from "lucide-react";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent } from "@/shared/ui/primitives/card";
import { Input } from "@/shared/ui/primitives/input";
import { Label } from "@/shared/ui/primitives/label";
import { AnimatedWordmark } from "@/shared/ui/animated-wordmark";
import { msg } from "@/shared/lib/messages";
import { tI18n } from "@/shared/lib/i18n";
import { cn } from "@/shared/lib/utils";
import { LoginHalo } from "./LoginHalo";

const ENTER_EASE = [0.16, 1, 0.3, 1] as const;

/**
 * Resolve where to send the user after login. next-auth's middleware appends a
 * ``callbackUrl`` query param when it bounces an unauthenticated request (e.g. a
 * ``/share/<token>`` link) to /login; honor it so the recipient lands back on
 * the page they came for. Only same-origin internal paths are accepted, so a
 * crafted ``callbackUrl`` can't turn login into an open redirect. Falls back to
 * the dashboard.
 */
function postLoginTarget(): string {
  if (typeof window === "undefined") return "/";
  const cb = new URLSearchParams(window.location.search).get("callbackUrl");
  if (!cb) return "/";
  try {
    const url = new URL(cb, window.location.origin);
    if (url.origin === window.location.origin) return url.pathname + url.search + url.hash;
  } catch {
    // Malformed callbackUrl — ignore and use the default.
  }
  return "/";
}

/**
 * Oversized SKYNET wordmark shared by every login state, so the SSO redirect
 * moment and the credential form read as the same place. It fills the column
 * width and morphs continuously as an ambient "alive" signal.
 */
function LoginHeader() {
  return (
    <div className="w-[min(90vw,520px)]">
      <AnimatedWordmark fluid autoMorph morphSpeed={250} />
    </div>
  );
}

/** Official multi-color Google "G", so the social button matches brand guidelines. */
function GoogleMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1Z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23Z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.1a6.6 6.6 0 0 1 0-4.2V7.06H2.18a11 11 0 0 0 0 9.88l3.66-2.84Z"
      />
      <path
        fill="#EA4335"
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1A11 11 0 0 0 2.18 7.06l3.66 2.84C6.71 7.3 9.14 5.38 12 5.38Z"
      />
    </svg>
  );
}

type AuthMode = "signin" | "signup";

export function LoginView() {
  const router = useRouter();
  const [mode, setMode] = useState<"loading" | "sso" | "ready">("loading");
  const [authMode, setAuthMode] = useState<AuthMode>("signin");
  const [oauth, setOauth] = useState<{ google: boolean; github: boolean }>({
    google: false,
    github: false,
  });
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    // If the providers endpoint errors (network blip, mis-deployed [...nextauth]
    // route), fall back to the credential form instead of hanging on the spinner.
    void getProviders()
      .then((providers) => {
        if (providers?.adfs) {
          setMode("sso");
          void signIn("adfs", { callbackUrl: postLoginTarget() });
          return;
        }
        setOauth({ google: !!providers?.google, github: !!providers?.github });
        setMode("ready");
      })
      .catch((err) => {
        console.warn("LoginView: getProviders failed", err);
        setMode("ready");
      });
  }, []);

  /**
   * Localize an error returned by /api/register. Backend semantic codes
   * (``accounts.*``, ``auth.not_configured``) resolve through the shared backend
   * catalog; anything else collapses to a generic "couldn't create the account".
   */
  function describeRegisterError(code: string): string {
    if (code.startsWith("accounts.") || code === "auth.not_configured") return tI18n(code);
    return msg("auth.login.register_failed");
  }

  function switchMode(next: AuthMode) {
    setAuthMode(next);
    setError("");
  }

  function handleOAuth(provider: "google" | "github") {
    setError("");
    void signIn(provider, { callbackUrl: postLoginTarget() });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const cleanEmail = email.trim().toLowerCase();
    if (!cleanEmail || !password) return;
    setError("");
    setLoading(true);
    try {
      if (authMode === "signup") {
        const res = await fetch("/api/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name.trim(), email: cleanEmail, password }),
        });
        if (!res.ok) {
          const data = (await res.json().catch(() => ({}))) as { error?: string };
          setError(describeRegisterError(data.error ?? "auth.login.register_failed"));
          setLoading(false);
          return;
        }
      }
      // Both paths finish by minting a session from the same credentials — signup
      // logs the new account straight in rather than bouncing back to the form.
      const result = await signIn("credentials", {
        email: cleanEmail,
        password,
        redirect: false,
      });
      if (result?.error) {
        setError(msg("auth.login.invalid_credentials"));
        setLoading(false);
        return;
      }
      // Soft-nav so we don't hard-reload and double-fetch the dashboard. Honor
      // the post-login target so a recipient bounced here from a /share/<token>
      // link lands back on it.
      router.push(postLoginTarget());
      router.refresh();
    } catch {
      setError(msg("auth.login.error"));
      setLoading(false);
    }
  }

  const isWorking = mode === "loading" || mode === "sso";
  const hasOAuth = oauth.google || oauth.github;
  const canSubmit = !!email.trim() && password.length > 0 && !loading;

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
        {isWorking ? (
          <div className="flex flex-col items-center">
            <LoginHeader />
            <div className="mt-9 flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              <span>{msg("auth.login.loading")}</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <LoginHeader />
            <Card className="mt-9 w-full">
              <CardContent className="px-6" dir="rtl">
                <div
                  role="tablist"
                  aria-label={msg("auth.login.form_aria")}
                  className="mb-5 flex rounded-lg bg-accent/60 p-1"
                >
                  {(["signin", "signup"] as const).map((tab) => (
                    <button
                      key={tab}
                      type="button"
                      role="tab"
                      aria-selected={authMode === tab}
                      onClick={() => switchMode(tab)}
                      className={cn(
                        "flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors duration-200",
                        authMode === tab
                          ? "bg-background text-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      {msg(tab === "signin" ? "auth.login.tab_signin" : "auth.login.tab_signup")}
                    </button>
                  ))}
                </div>

                {hasOAuth && (
                  <>
                    <div className="space-y-2.5">
                      {oauth.google && (
                        <Button
                          type="button"
                          variant="outline"
                          size="lg"
                          onClick={() => handleOAuth("google")}
                          className="h-11 w-full gap-2.5 text-[0.9375rem] font-medium"
                        >
                          <GoogleMark className="size-[18px]" />
                          {msg("auth.login.with_google")}
                        </Button>
                      )}
                      {oauth.github && (
                        <Button
                          type="button"
                          variant="outline"
                          size="lg"
                          onClick={() => handleOAuth("github")}
                          className="h-11 w-full gap-2.5 text-[0.9375rem] font-medium"
                        >
                          <Github className="size-[18px]" />
                          {msg("auth.login.with_github")}
                        </Button>
                      )}
                    </div>
                    <div className="my-5 flex items-center gap-3" aria-hidden="true">
                      <span className="h-px flex-1 bg-border" />
                      <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {msg("auth.login.divider")}
                      </span>
                      <span className="h-px flex-1 bg-border" />
                    </div>
                  </>
                )}

                <form onSubmit={handleSubmit} className="space-y-3.5">
                  <AnimatePresence initial={false}>
                    {authMode === "signup" && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <Label
                          htmlFor="login-name"
                          className="mb-1.5 block text-xs font-medium text-muted-foreground"
                        >
                          {msg("auth.login.name")}
                        </Label>
                        <Input
                          id="login-name"
                          value={name}
                          onChange={(e) => setName(e.target.value)}
                          placeholder={msg("auth.login.name_placeholder")}
                          autoComplete="name"
                          dir="auto"
                          className="h-11"
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <div>
                    <Label
                      htmlFor="login-email"
                      className="mb-1.5 block text-xs font-medium text-muted-foreground"
                    >
                      {msg("auth.login.email")}
                    </Label>
                    <Input
                      id="login-email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder={msg("auth.login.email_placeholder")}
                      autoFocus
                      autoComplete="email"
                      dir="ltr"
                      className="h-11 text-left"
                    />
                  </div>

                  <div>
                    <Label
                      htmlFor="login-password"
                      className="mb-1.5 block text-xs font-medium text-muted-foreground"
                    >
                      {msg("auth.login.password")}
                    </Label>
                    <Input
                      id="login-password"
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder={msg("auth.login.password_placeholder")}
                      autoComplete={authMode === "signup" ? "new-password" : "current-password"}
                      dir="ltr"
                      className="h-11 text-left"
                    />
                    {authMode === "signup" && (
                      <p className="mt-1.5 text-xs text-muted-foreground">
                        {msg("auth.login.password_hint")}
                      </p>
                    )}
                  </div>

                  <AnimatePresence>
                    {error && (
                      <motion.p
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.2 }}
                        className="text-sm text-destructive"
                        role="alert"
                      >
                        {error}
                      </motion.p>
                    )}
                  </AnimatePresence>

                  <Button
                    type="submit"
                    size="lg"
                    disabled={!canSubmit}
                    className="h-11 w-full gap-2 text-[0.9375rem] font-medium"
                  >
                    {loading && <Loader2 className="size-4 animate-spin" />}
                    {msg(
                      authMode === "signin"
                        ? "auth.login.signin_submit"
                        : "auth.login.signup_submit",
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        )}
      </motion.div>
    </div>
  );
}
