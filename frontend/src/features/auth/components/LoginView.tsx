"use client";

import { useEffect, useState } from "react";
import { signIn, getProviders, getSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { CircleNotch, GithubLogo, Fingerprint, ArrowLeft } from "@/shared/ui/icons";
import {
  browserSupportsWebAuthn,
  browserSupportsWebAuthnAutofill,
  platformAuthenticatorIsAvailable,
  startAuthentication,
  startRegistration,
  WebAuthnAbortService,
} from "@simplewebauthn/browser";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent } from "@/shared/ui/primitives/card";
import { Input } from "@/shared/ui/primitives/input";
import { Label } from "@/shared/ui/primitives/label";
import { AnimatedWordmark } from "@/shared/ui/animated-wordmark";
import { LanguageSwitcher } from "@/shared/ui/language-switcher";
import { msg } from "@/shared/lib/messages";
import {
  getPasskeyRegistrationOptions,
  getSecurityStatus,
  registerPasskey,
  setApiAuthToken,
} from "@/shared/lib/api";
import { tI18n } from "@/shared/lib/i18n";
import { cn } from "@/shared/lib/utils";
import { track, TelemetryEvent } from "@/shared/lib/telemetry";
import { LEGAL_LINKS } from "@/features/legal/legal-config";
import { LoginHalo } from "./LoginHalo";

const ENTER_EASE = [0.16, 1, 0.3, 1] as const;

// Mirrors the backend's password_policy (the accounts.* 422s); the
// common-password blocklist stays server-side and surfaces through
// describeRegisterError like any other accounts.* code.
const MIN_PASSWORD_LENGTH = 8;
const MAX_PASSWORD_LENGTH = 128;
const MIN_EMAIL_LOCAL_PART = 4;

type PasswordRuleKey =
  | "auth.login.password_hint"
  | "auth.login.password_rule_max"
  | "auth.login.password_rule_email";

/** The policy rules the typed password does not yet satisfy, for live feedback. */
function unmetPasswordRules(password: string, email: string): PasswordRuleKey[] {
  const rules: PasswordRuleKey[] = [];
  if (password.length < MIN_PASSWORD_LENGTH) rules.push("auth.login.password_hint");
  if (password.length > MAX_PASSWORD_LENGTH) rules.push("auth.login.password_rule_max");
  const localPart = email.trim().toLowerCase().split("@", 1)[0] ?? "";
  if (localPart.length >= MIN_EMAIL_LOCAL_PART && password.toLowerCase().includes(localPart)) {
    rules.push("auth.login.password_rule_email");
  }
  return rules;
}

const TWOFA_LINK_CLASS =
  "block cursor-pointer text-xs font-medium text-muted-foreground underline-offset-2 transition-colors hover:text-foreground hover:underline";

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
 * OAuth round-trips leave this page entirely, so there is no moment to pause
 * on the enrollment offer the way password sign-ins do. Route the provider's
 * success redirect back through /login with a marker; the mount effect spots
 * it, runs the same one-time offer, and then continues to the real target
 * (still carried in ``callbackUrl``).
 */
function oauthReturnUrl(): string {
  const target = postLoginTarget();
  const params = new URLSearchParams({ passkey_offer: "1" });
  if (target !== "/") params.set("callbackUrl", target);
  return `/login?${params.toString()}`;
}

/**
 * Oversized SKYNET wordmark shared by every login state, so the SSO redirect
 * moment and the credential form read as the same place. It fills the column
 * width and morphs continuously as an ambient "alive" signal.
 */
function LoginHeader() {
  return (
    <div className="w-[min(90vw,520px)]">
      <AnimatedWordmark fluid autoMorph autoMorphDuration={10000} morphSpeed={250} />
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

type TwoFactorMethod = "totp" | "email" | "recovery";

interface TwoFactorState {
  /** Methods the backend reported as usable ("totp" / "email"). */
  methods: string[];
  /** The entry mode the code input currently targets. */
  mode: TwoFactorMethod;
  /** Whether an emailed code was already requested this attempt. */
  emailSent: boolean;
}

interface ResetState {
  /** Which step of the two-step forgot-password flow is showing. */
  step: "request" | "confirm";
}

export function LoginView() {
  const router = useRouter();
  const prefersReduced = useReducedMotion();
  const [mode, setMode] = useState<"loading" | "sso" | "ready">("loading");
  const [authMode, setAuthMode] = useState<AuthMode>("signin");
  const [oauth, setOauth] = useState<{ google: boolean; github: boolean }>({
    google: false,
    github: false,
  });
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Reveal "Forgot password?" only after a password was rejected — a wrong
  // password is the one moment the link is actually useful, so it stays hidden
  // otherwise instead of nudging every visitor toward a reset.
  const [badCredentials, setBadCredentials] = useState(false);
  const [twoFactor, setTwoFactor] = useState<TwoFactorState | null>(null);
  const [twoFactorCode, setTwoFactorCode] = useState("");
  const [sendingCode, setSendingCode] = useState(false);
  const [passkeyLoading, setPasskeyLoading] = useState(false);
  const [passkeySupported, setPasskeySupported] = useState(false);
  const [passkeyOffer, setPasskeyOffer] = useState<"offer" | "saving" | null>(null);
  const [reset, setReset] = useState<ResetState | null>(null);
  const [resetEmail, setResetEmail] = useState("");
  const [resetCode, setResetCode] = useState("");
  const [resetNewPassword, setResetNewPassword] = useState("");
  const [resetLoading, setResetLoading] = useState(false);
  const [verify, setVerify] = useState<{ email: string } | null>(null);
  const [verifyCode, setVerifyCode] = useState("");
  const [verifyLoading, setVerifyLoading] = useState(false);

  useEffect(() => {
    setPasskeySupported(browserSupportsWebAuthn());
  }, []);

  useEffect(() => {
    if (mode !== "ready" || authMode !== "signin" || !passkeySupported) return;
    if (twoFactor || passkeyOffer || reset || verify) return;
    let cancelled = false;
    void (async () => {
      try {
        if (!(await browserSupportsWebAuthnAutofill())) return;
        const optionsRes = await fetch("/api/webauthn/options", { method: "POST" });
        if (!optionsRes.ok || cancelled) return;
        const options = await optionsRes.json();
        const assertion = await startAuthentication(options, true);
        if (cancelled) return;
        const result = await signIn("passkey", {
          assertion: JSON.stringify(assertion),
          redirect: false,
        });
        if (cancelled) return;
        if (!result?.error) {
          track(TelemetryEvent.LoginSucceeded, { method: "passkey_conditional" });
          finishLogin();
        }
      } catch (err) {
        const name = (err as Error)?.name;
        if (name === "AbortError" || name === "NotAllowedError") return;
        const cause = (err as { cause?: { name?: string } })?.cause?.name;
        if (cause === "AbortError" || cause === "NotAllowedError") return;
      }
    })();
    return () => {
      cancelled = true;
      WebAuthnAbortService.cancelCeremony();
    };
  }, [mode, authMode, passkeySupported, twoFactor, passkeyOffer, reset, verify]);

  useEffect(() => {
    // If the providers endpoint errors (network blip, mis-deployed [...nextauth]
    // route), fall back to the credential form instead of hanging on the spinner.
    const loadProviders = () =>
      getProviders()
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
    // Back from the OAuth round-trip (?passkey_offer=1): keep the spinner up
    // and run the same enrollment offer password sign-ins get. Arriving here
    // without a session (cancelled consent, provider error) falls through to
    // the normal form.
    const returning =
      new URLSearchParams(window.location.search).get("passkey_offer") === "1";
    if (!returning) {
      void loadProviders();
      return;
    }
    void getSession().then((session) => {
      if (session?.backendAccessToken) return offerPasskeyOrFinish();
      return loadProviders();
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

  /**
   * Localize an error returned by the password-reset proxies. Backend semantic
   * codes (``accounts.*``, ``auth.not_configured``) resolve through the shared
   * catalog; anything else collapses to a generic "couldn't reset your password".
   */
  function describeResetError(code: string): string {
    if (code.startsWith("accounts.") || code === "auth.not_configured") return tI18n(code);
    return msg("auth.login.reset_failed");
  }

  /**
   * Localize an error from the email-verification proxies. Backend semantic
   * codes (``accounts.*``, ``auth.not_configured``) resolve through the shared
   * catalog; anything else collapses to a generic "couldn't verify your email".
   */
  function describeVerifyError(code: string): string {
    if (code.startsWith("accounts.") || code === "auth.not_configured") return tI18n(code);
    return msg("auth.login.verify_failed");
  }

  /** Open the forgot-password flow, seeding it with any email already typed. */
  function enterReset() {
    setReset({ step: "request" });
    setResetEmail(email.trim().toLowerCase());
    setResetCode("");
    setResetNewPassword("");
    setError("");
  }

  function leaveReset() {
    setReset(null);
    setError("");
  }

  /**
   * Ask the backend to email a reset code. The response is deliberately identical
   * for known and unknown addresses, so a success only advances the card to the
   * code step; it never reveals whether the email has an account. Doubles as the
   * "resend" action on the code step.
   */
  async function requestResetCode(): Promise<void> {
    const cleanEmail = resetEmail.trim().toLowerCase();
    if (!cleanEmail) return;
    setError("");
    setResetLoading(true);
    try {
      const res = await fetch("/api/password-reset/request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: cleanEmail }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setError(describeResetError(data.error ?? "auth.login.reset_failed"));
        return;
      }
      setReset({ step: "confirm" });
      setResetCode("");
    } catch {
      setError(msg("auth.login.error"));
    } finally {
      setResetLoading(false);
    }
  }

  function handleResetRequest(e: React.FormEvent) {
    e.preventDefault();
    void requestResetCode();
  }

  /**
   * Verify the emailed code, set the new password, then sign in with it. A
   * 2FA-protected account still answers the fresh password with a "code required"
   * signal, so the same second-factor step the normal sign-in uses is reused
   * here rather than letting a reset slip past 2FA.
   */
  async function handleResetConfirm(e: React.FormEvent) {
    e.preventDefault();
    const cleanEmail = resetEmail.trim().toLowerCase();
    const code = resetCode.trim();
    if (!cleanEmail || !code || !resetNewPassword) return;
    setError("");
    setResetLoading(true);
    try {
      const res = await fetch("/api/password-reset/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: cleanEmail, code, new_password: resetNewPassword }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setError(describeResetError(data.error ?? "auth.login.reset_failed"));
        setResetLoading(false);
        return;
      }
      // Password updated — mint a session from the new credentials the same way a
      // fresh signup does, pivoting to the 2FA step if the account has one.
      setEmail(cleanEmail);
      setPassword(resetNewPassword);
      const result = await signIn("credentials", {
        email: cleanEmail,
        password: resetNewPassword,
        redirect: false,
      });
      if (result?.error) {
        const signal = result.code ?? "";
        if (signal.startsWith("2fa_required:")) {
          const methods = signal.slice("2fa_required:".length).split(" ").filter(Boolean);
          setReset(null);
          setTwoFactor({
            methods,
            mode: methods.includes("totp") ? "totp" : "email",
            emailSent: false,
          });
          setTwoFactorCode("");
          setResetLoading(false);
          return;
        }
        setReset(null);
        setError(msg("auth.login.invalid_credentials"));
        setResetLoading(false);
        return;
      }
      setReset(null);
      track(TelemetryEvent.LoginSucceeded, { method: "password_reset" });
      await offerPasskeyOrFinish();
    } catch {
      setError(msg("auth.login.error"));
      setResetLoading(false);
    }
  }

  function leaveVerify() {
    setVerify(null);
    setVerifyCode("");
    setError("");
  }

  /**
   * Confirm the emailed code, then sign in with the password still held in
   * state. A 2FA-protected account answers the now-verified credentials with a
   * "code required" signal, so the same second-factor step the normal sign-in
   * uses is reused here rather than letting verification slip past 2FA.
   */
  async function handleVerifyConfirm(e: React.FormEvent) {
    e.preventDefault();
    if (!verify) return;
    const cleanEmail = verify.email;
    const code = verifyCode.trim();
    if (!cleanEmail || !code) return;
    setError("");
    setVerifyLoading(true);
    try {
      const res = await fetch("/api/email-verify/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: cleanEmail, code }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setError(describeVerifyError(data.error ?? "auth.login.verify_failed"));
        setVerifyLoading(false);
        return;
      }
      const result = await signIn("credentials", {
        email: cleanEmail,
        password,
        redirect: false,
      });
      if (result?.error) {
        const signal = result.code ?? "";
        if (signal.startsWith("2fa_required:")) {
          const methods = signal.slice("2fa_required:".length).split(" ").filter(Boolean);
          setVerify(null);
          setTwoFactor({
            methods,
            mode: methods.includes("totp") ? "totp" : "email",
            emailSent: false,
          });
          setTwoFactorCode("");
          setVerifyLoading(false);
          return;
        }
        setVerify(null);
        setError(msg("auth.login.invalid_credentials"));
        setVerifyLoading(false);
        return;
      }
      setVerify(null);
      track(TelemetryEvent.LoginSucceeded, { method: "email_verified" });
      await offerPasskeyOrFinish();
    } catch {
      setError(msg("auth.login.error"));
      setVerifyLoading(false);
    }
  }

  /**
   * Re-send the confirmation code. The backend acknowledges identically for
   * unknown, already-verified, and on-cooldown addresses, so a success only
   * clears any error; it never reveals the account's verification state.
   */
  async function resendVerifyCode(): Promise<void> {
    if (!verify) return;
    setError("");
    setVerifyLoading(true);
    try {
      const res = await fetch("/api/email-verify/request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: verify.email }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setError(describeVerifyError(data.error ?? "auth.login.verify_failed"));
        return;
      }
      setVerifyCode("");
    } catch {
      setError(msg("auth.login.error"));
    } finally {
      setVerifyLoading(false);
    }
  }

  function switchMode(next: AuthMode) {
    setAuthMode(next);
    setError("");
    setBadCredentials(false);
  }

  function handleOAuth(provider: "google" | "github") {
    setError("");
    void signIn(provider, { callbackUrl: oauthReturnUrl() });
  }

  /**
   * Soft-nav to the post-login target so we don't hard-reload and double-fetch
   * the dashboard, and a recipient bounced here from a /share/<token> link
   * lands back on it.
   */
  function finishLogin() {
    router.push(postLoginTarget());
    router.refresh();
  }

  /**
   * After a password sign-in on hardware with Face ID / Touch ID, pause once to
   * offer creating a passkey here. WebAuthn can only enroll an authenticated
   * user, so the "sign in with a passkey" button alone can never reach the
   * platform authenticator on a device with nothing stored — the browser falls
   * back to its cross-device QR sheet. Any hiccup while asking the question
   * (no platform authenticator, the status probe failing, a passkey already
   * registered) falls through to the normal redirect.
   */
  async function offerPasskeyOrFinish() {
    try {
      if (!browserSupportsWebAuthn() || !(await platformAuthenticatorIsAvailable())) {
        finishLogin();
        return;
      }
      // The session broadcast that normally feeds the api layer its bearer is
      // still in flight this soon after signIn — mint the token directly.
      const session = await getSession();
      if (!session?.backendAccessToken) {
        finishLogin();
        return;
      }
      setApiAuthToken(session.backendAccessToken);
      const status = await getSecurityStatus();
      if (status.passkeys.length > 0) {
        finishLogin();
        return;
      }
      setLoading(false);
      // The OAuth-return path arrives with the card still on its spinner.
      setMode("ready");
      setPasskeyOffer("offer");
    } catch {
      finishLogin();
    }
  }

  /**
   * Run the browser's platform-authenticator enrollment (Face ID / Touch ID
   * sheet) and store the credential. A genuine dismissal continues into the app
   * — enrollment is optional. But a failure *before* any sheet appears (a stale
   * bearer, the options call erroring) used to be swallowed too, silently
   * dropping the user into the app as if they had declined; that reads as "the
   * button does nothing". Those failures now surface a message and keep the
   * offer so the user can retry.
   */
  async function enrollPasskey() {
    setError("");
    setPasskeyOffer("saving");
    try {
      // The bearer minted right after sign-in is short-lived and the login page
      // registers no 401 refresher, so re-mint from the current session before
      // the authenticated options call — otherwise a stale token 401s the fetch
      // and the whole ceremony collapses before any prompt shows.
      const session = await getSession();
      if (session?.backendAccessToken) {
        setApiAuthToken(session.backendAccessToken);
      }
      const options = await getPasskeyRegistrationOptions();
      const credential = await startRegistration(
        options as unknown as Parameters<typeof startRegistration>[0],
      );
      await registerPasskey(credential, "");
      finishLogin();
    } catch (err) {
      const name = (err as Error)?.name;
      const cause = (err as { cause?: { name?: string } })?.cause?.name;
      const dismissed =
        name === "NotAllowedError" ||
        cause === "NotAllowedError" ||
        name === "AbortError" ||
        cause === "AbortError";
      if (dismissed) {
        // The user closed the Face ID / Touch ID sheet, or a newer ceremony
        // superseded this one — enrolling is optional, so continue.
        finishLogin();
        return;
      }
      setError(msg("auth.login.passkey_enroll_failed"));
      setPasskeyOffer("offer");
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const cleanEmail = email.trim().toLowerCase();
    if (!cleanEmail || !password) return;
    setError("");
    setBadCredentials(false);
    setLoading(true);
    try {
      if (authMode === "signup") {
        track(TelemetryEvent.SignupStarted);
        const res = await fetch("/api/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: cleanEmail, password }),
        });
        if (!res.ok) {
          const data = (await res.json().catch(() => ({}))) as { error?: string };
          track(TelemetryEvent.LoginFailed, { method: "signup" });
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
        // A 2FA-protected account answers the correct password with a typed
        // "code required" signal carrying its usable methods — pivot the card
        // to the verification step instead of surfacing an error.
        const code = result.code ?? "";
        if (code.startsWith("2fa_required:")) {
          const methods = code.slice("2fa_required:".length).split(" ").filter(Boolean);
          setTwoFactor({
            methods,
            mode: methods.includes("totp") ? "totp" : "email",
            emailSent: false,
          });
          setTwoFactorCode("");
          setLoading(false);
          return;
        }
        // An unverified account (email delivery is configured, the address is
        // not yet confirmed) answers the correct password with this signal —
        // pivot the card to the email-confirmation step, seeded with the
        // address just entered, instead of showing a dead-end error.
        if (code === "email_unverified") {
          setVerify({ email: cleanEmail });
          setVerifyCode("");
          setLoading(false);
          return;
        }
        track(TelemetryEvent.LoginFailed, {
          method: authMode === "signup" ? "signup" : "credentials",
        });
        setError(msg("auth.login.invalid_credentials"));
        // A wrong password is the one case where offering a reset helps, so this
        // is where "Forgot password?" earns its place on the card.
        if (authMode === "signin") {
          setBadCredentials(true);
        }
        setLoading(false);
        return;
      }
      if (authMode === "signup") {
        track(TelemetryEvent.SignupSucceeded);
      } else {
        track(TelemetryEvent.LoginSucceeded, { method: "credentials" });
      }
      await offerPasskeyOrFinish();
    } catch {
      setError(msg("auth.login.error"));
      setLoading(false);
    }
  }

  async function handleTwoFactorSubmit(e: React.FormEvent) {
    e.preventDefault();
    const code = twoFactorCode.trim();
    if (!twoFactor || !code) return;
    setError("");
    setLoading(true);
    try {
      const codeField =
        twoFactor.mode === "totp"
          ? { totpCode: code }
          : twoFactor.mode === "email"
            ? { emailCode: code }
            : { recoveryCode: code };
      const result = await signIn("credentials", {
        email: email.trim().toLowerCase(),
        password,
        ...codeField,
        redirect: false,
      });
      if (result?.error) {
        track(TelemetryEvent.LoginFailed, { method: "credentials_2fa" });
        setError(
          (result.code ?? "").startsWith("2fa")
            ? msg("auth.login.twofa_invalid")
            : msg("auth.login.invalid_credentials"),
        );
        setLoading(false);
        return;
      }
      track(TelemetryEvent.LoginSucceeded, { method: "credentials_2fa" });
      await offerPasskeyOrFinish();
    } catch {
      setError(msg("auth.login.error"));
      setLoading(false);
    }
  }

  async function sendEmailCode() {
    if (!twoFactor) return;
    setError("");
    setSendingCode(true);
    try {
      const res = await fetch("/api/2fa/email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
      });
      if (!res.ok) {
        setError(msg("auth.login.twofa_send_failed"));
        return;
      }
      setTwoFactor({ ...twoFactor, mode: "email", emailSent: true });
      setTwoFactorCode("");
    } catch {
      setError(msg("auth.login.twofa_send_failed"));
    } finally {
      setSendingCode(false);
    }
  }

  function switchTwoFactorMode(nextMode: TwoFactorMethod) {
    if (!twoFactor) return;
    setTwoFactor({ ...twoFactor, mode: nextMode });
    setTwoFactorCode("");
    setError("");
  }

  function leaveTwoFactor() {
    setTwoFactor(null);
    setTwoFactorCode("");
    setError("");
  }

  async function handlePasskey() {
    setError("");
    // A passkey attempt is a separate flow from the password form; drop any
    // "wrong password" state so its "Forgot password?" link and error never
    // linger under a passkey message like "No passkey found on this device".
    setBadCredentials(false);
    setPasskeyLoading(true);
    WebAuthnAbortService.cancelCeremony();
    try {
      const optionsRes = await fetch("/api/webauthn/options", { method: "POST" });
      if (!optionsRes.ok) {
        setError(msg("auth.login.passkey_failed"));
        return;
      }
      const options = await optionsRes.json();
      const assertion = await startAuthentication(options);
      const result = await signIn("passkey", {
        assertion: JSON.stringify(assertion),
        redirect: false,
      });
      if (result?.error) {
        track(TelemetryEvent.LoginFailed, { method: "passkey" });
        setError(msg("auth.login.passkey_failed"));
        return;
      }
      track(TelemetryEvent.LoginSucceeded, { method: "passkey" });
      finishLogin();
    } catch (err) {
      const name = (err as Error)?.name;
      const cause = (err as { cause?: { name?: string } })?.cause?.name;
      const aborted = name === "AbortError" || cause === "AbortError";
      const notAllowed = name === "NotAllowedError" || cause === "NotAllowedError";
      if (aborted) {
        // Programmatic abort (navigating away, or a superseding ceremony) — not
        // something the user did, so it stays silent.
      } else if (notAllowed) {
        // The authenticator offered no passkey for this site, or the prompt was
        // dismissed. WebAuthn collapses both into NotAllowedError, so rather than
        // leave the button a silent dead-end, point the user at the only path
        // that makes passkey sign-in possible: sign in another way, then enroll.
        setError(msg("auth.login.passkey_none"));
      } else {
        setError(msg("auth.login.passkey_failed"));
      }
    } finally {
      setPasskeyLoading(false);
    }
  }

  const isWorking = mode === "loading" || mode === "sso";
  // OAuth and passkeys are sign-IN affordances only: "continue with Google"
  // or "sign in with a passkey" make no sense under the "create an account"
  // tab, so both show on the signin tab exclusively.
  const showOAuth = (oauth.google || oauth.github) && authMode === "signin";
  const showPasskey = passkeySupported && authMode === "signin";
  const canSubmit = !loading && !!email.trim() && password.length > 0;
  const passwordRules =
    authMode === "signup" && password.length > 0 ? unmetPasswordRules(password, email) : [];
  const resetPasswordRules =
    reset?.step === "confirm" && resetNewPassword.length > 0
      ? unmetPasswordRules(resetNewPassword, resetEmail)
      : [];

  return (
    <div className="relative flex min-h-dvh w-full items-center justify-center px-4 py-10">
      <LanguageSwitcher className="absolute end-4 top-4 z-20 bg-background/70 backdrop-blur-sm" />
      <LoginHalo />
      {/* Clear the full-height centre column so the wordmark and the (taller)
          sign-up form never collide with a halo chip. A centered radial can't
          keep both the top logo and a long form clean at once, so this is a
          vertical band: opaque only across the form's column (30–70%), going
          fully clear by ~24%/76% so the wing chips that hug the form on either
          side read crisply instead of dissolving into the mask. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-[1]"
        style={{
          background:
            "linear-gradient(90deg, rgba(250,248,245,0) 0%, rgba(250,248,245,0) 24%, rgba(250,248,245,1) 30%, rgba(250,248,245,1) 70%, rgba(250,248,245,0) 76%, rgba(250,248,245,0) 100%)",
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
              <CircleNotch className="size-4 animate-spin" />
              <span>{msg("auth.login.loading")}</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <LoginHeader />
            <Card className="mt-9 w-full">
              <CardContent className="px-6">
                {passkeyOffer ? (
                  <div className="flex flex-col items-center py-2 text-center">
                    <div className="flex size-12 items-center justify-center rounded-full bg-accent">
                      <Fingerprint className="size-6 text-foreground" aria-hidden="true" />
                    </div>
                    <p className="mt-4 text-sm font-semibold text-foreground">
                      {msg("auth.login.passkey_offer_title")}
                    </p>
                    <p className="mt-1 max-w-[36ch] text-xs leading-relaxed text-muted-foreground">
                      {msg("auth.login.passkey_offer_description")}
                    </p>
                    <AnimatePresence>
                      {error && (
                        <motion.p
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.2 }}
                          className="mt-3 text-sm text-destructive"
                          role="alert"
                        >
                          {error}
                        </motion.p>
                      )}
                    </AnimatePresence>
                    <Button
                      type="button"
                      size="lg"
                      disabled={passkeyOffer === "saving"}
                      onClick={() => void enrollPasskey()}
                      className="mt-5 h-11 w-full gap-2 text-[0.9375rem] font-medium"
                    >
                      {passkeyOffer === "saving" ? (
                        <CircleNotch className="size-4 animate-spin" />
                      ) : (
                        <Fingerprint className="size-[18px]" />
                      )}
                      {msg("auth.login.passkey_offer_accept")}
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="lg"
                      disabled={passkeyOffer === "saving"}
                      onClick={finishLogin}
                      className="mt-2 h-11 w-full text-[0.9375rem] font-medium text-muted-foreground"
                    >
                      {msg("auth.login.passkey_offer_skip")}
                    </Button>
                  </div>
                ) : twoFactor ? (
                  <div>
                    <button
                      type="button"
                      onClick={leaveTwoFactor}
                      className="mb-4 flex cursor-pointer items-center gap-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <ArrowLeft className="size-3.5 rtl:-scale-x-100" aria-hidden="true" />
                      {msg("auth.login.twofa_back")}
                    </button>
                    <p className="text-sm font-semibold text-foreground">
                      {msg("auth.login.twofa_heading")}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {twoFactor.mode === "totp"
                        ? msg("auth.login.twofa_totp_hint")
                        : twoFactor.mode === "email"
                          ? twoFactor.emailSent
                            ? msg("auth.login.twofa_email_sent")
                            : msg("auth.login.twofa_email_hint")
                          : msg("auth.login.twofa_recovery_hint")}
                    </p>
                    {twoFactor.mode === "email" && (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        disabled={sendingCode}
                        onClick={() => void sendEmailCode()}
                        className="mt-3 gap-2"
                      >
                        {sendingCode && <CircleNotch className="size-3.5 animate-spin" />}
                        {msg(
                          twoFactor.emailSent ? "auth.login.twofa_resend" : "auth.login.twofa_send",
                        )}
                      </Button>
                    )}
                    <form onSubmit={handleTwoFactorSubmit} className="mt-4 space-y-3.5">
                      <div>
                        <Label
                          htmlFor="twofa-code"
                          className="mb-1.5 block text-xs font-medium text-muted-foreground"
                        >
                          {msg("auth.login.twofa_code_label")}
                        </Label>
                        <Input
                          id="twofa-code"
                          value={twoFactorCode}
                          onChange={(e) => setTwoFactorCode(e.target.value)}
                          placeholder={twoFactor.mode === "recovery" ? "XXXX-XXXX" : "123456"}
                          autoFocus
                          autoComplete="one-time-code"
                          inputMode={twoFactor.mode === "recovery" ? "text" : "numeric"}
                          dir="ltr"
                          className="h-11 text-left"
                        />
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
                        disabled={loading || !twoFactorCode.trim()}
                        className="h-11 w-full gap-2 text-[0.9375rem] font-medium"
                      >
                        {loading && <CircleNotch className="size-4 animate-spin" />}
                        {msg("auth.login.twofa_verify")}
                      </Button>
                    </form>
                    <div className="mt-4 space-y-1.5">
                      {twoFactor.mode !== "totp" && twoFactor.methods.includes("totp") && (
                        <button
                          type="button"
                          onClick={() => switchTwoFactorMode("totp")}
                          className={TWOFA_LINK_CLASS}
                        >
                          {msg("auth.login.twofa_use_totp")}
                        </button>
                      )}
                      {twoFactor.mode !== "email" && twoFactor.methods.includes("email") && (
                        <button
                          type="button"
                          onClick={() => switchTwoFactorMode("email")}
                          className={TWOFA_LINK_CLASS}
                        >
                          {msg("auth.login.twofa_use_email")}
                        </button>
                      )}
                      {twoFactor.mode !== "recovery" && (
                        <button
                          type="button"
                          onClick={() => switchTwoFactorMode("recovery")}
                          className={TWOFA_LINK_CLASS}
                        >
                          {msg("auth.login.twofa_use_recovery")}
                        </button>
                      )}
                    </div>
                  </div>
                ) : reset ? (
                  <div>
                    <button
                      type="button"
                      onClick={leaveReset}
                      className="mb-4 flex cursor-pointer items-center gap-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <ArrowLeft className="size-3.5 rtl:-scale-x-100" aria-hidden="true" />
                      {msg("auth.login.twofa_back")}
                    </button>
                    <p className="text-sm font-semibold text-foreground">
                      {msg("auth.login.reset_heading")}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {reset.step === "request"
                        ? msg("auth.login.reset_request_hint")
                        : msg("auth.login.reset_confirm_hint")}
                    </p>
                    {reset.step === "request" ? (
                      <form onSubmit={handleResetRequest} className="mt-4 space-y-3.5">
                        <div>
                          <Label
                            htmlFor="reset-email"
                            className="mb-1.5 block text-xs font-medium text-muted-foreground"
                          >
                            {msg("auth.login.email")}
                          </Label>
                          <Input
                            id="reset-email"
                            type="email"
                            value={resetEmail}
                            onChange={(e) => setResetEmail(e.target.value)}
                            placeholder={msg("auth.login.email_placeholder")}
                            autoFocus
                            autoComplete="username"
                            dir="ltr"
                            className="h-11 text-left"
                          />
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
                          disabled={resetLoading || !resetEmail.trim()}
                          className="h-11 w-full gap-2 text-[0.9375rem] font-medium"
                        >
                          {resetLoading && <CircleNotch className="size-4 animate-spin" />}
                          {msg("auth.login.reset_send")}
                        </Button>
                      </form>
                    ) : (
                      <form onSubmit={handleResetConfirm} className="mt-4 space-y-3.5">
                        <div>
                          <Label
                            htmlFor="reset-code"
                            className="mb-1.5 block text-xs font-medium text-muted-foreground"
                          >
                            {msg("auth.login.reset_code_label")}
                          </Label>
                          <Input
                            id="reset-code"
                            value={resetCode}
                            onChange={(e) => setResetCode(e.target.value)}
                            placeholder="123456"
                            autoFocus
                            autoComplete="one-time-code"
                            inputMode="numeric"
                            dir="ltr"
                            className="h-11 text-left"
                          />
                        </div>
                        <div>
                          <Label
                            htmlFor="reset-password"
                            className="mb-1.5 block text-xs font-medium text-muted-foreground"
                          >
                            {msg("auth.login.reset_new_password")}
                          </Label>
                          <Input
                            id="reset-password"
                            type="password"
                            value={resetNewPassword}
                            onChange={(e) => setResetNewPassword(e.target.value)}
                            placeholder={msg("auth.login.password_placeholder")}
                            autoComplete="new-password"
                            dir="ltr"
                            className="h-11 text-left"
                          />
                          <AnimatePresence initial={false}>
                            {resetPasswordRules.map((rule) => (
                              <motion.p
                                key={rule}
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: "auto" }}
                                exit={{ opacity: 0, height: 0 }}
                                transition={{ duration: 0.2 }}
                                className="mt-1.5 text-xs text-muted-foreground"
                              >
                                {msg(rule)}
                              </motion.p>
                            ))}
                          </AnimatePresence>
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
                          disabled={resetLoading || !resetCode.trim() || !resetNewPassword}
                          className="h-11 w-full gap-2 text-[0.9375rem] font-medium"
                        >
                          {resetLoading && <CircleNotch className="size-4 animate-spin" />}
                          {msg("auth.login.reset_submit")}
                        </Button>
                        <button
                          type="button"
                          onClick={() => void requestResetCode()}
                          disabled={resetLoading}
                          className={TWOFA_LINK_CLASS}
                        >
                          {msg("auth.login.twofa_resend")}
                        </button>
                      </form>
                    )}
                  </div>
                ) : verify ? (
                  <div>
                    <button
                      type="button"
                      onClick={leaveVerify}
                      className="mb-4 flex cursor-pointer items-center gap-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <ArrowLeft className="size-3.5 rtl:-scale-x-100" aria-hidden="true" />
                      {msg("auth.login.twofa_back")}
                    </button>
                    <p className="text-sm font-semibold text-foreground">
                      {msg("auth.login.verify_heading")}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {msg("auth.login.verify_hint")}
                    </p>
                    <form onSubmit={handleVerifyConfirm} className="mt-4 space-y-3.5">
                      <div>
                        <Label
                          htmlFor="verify-code"
                          className="mb-1.5 block text-xs font-medium text-muted-foreground"
                        >
                          {msg("auth.login.verify_code_label")}
                        </Label>
                        <Input
                          id="verify-code"
                          value={verifyCode}
                          onChange={(e) => setVerifyCode(e.target.value)}
                          placeholder="123456"
                          autoFocus
                          autoComplete="one-time-code"
                          inputMode="numeric"
                          dir="ltr"
                          className="h-11 text-left"
                        />
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
                        disabled={verifyLoading || !verifyCode.trim()}
                        className="h-11 w-full gap-2 text-[0.9375rem] font-medium"
                      >
                        {verifyLoading && <CircleNotch className="size-4 animate-spin" />}
                        {msg("auth.login.verify_submit")}
                      </Button>
                      <button
                        type="button"
                        onClick={() => void resendVerifyCode()}
                        disabled={verifyLoading}
                        className={TWOFA_LINK_CLASS}
                      >
                        {msg("auth.login.twofa_resend")}
                      </button>
                    </form>
                  </div>
                ) : (
                  <>
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
                            "relative flex-1 cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors duration-200",
                            authMode === tab
                              ? "text-foreground"
                              : "text-muted-foreground hover:text-foreground",
                          )}
                        >
                          {/* Shared-layoutId pill: the same id on whichever tab is
                              active lets Framer SLIDE the highlight between the two
                              instead of snapping, mirroring the settings rail. */}
                          {authMode === tab && (
                            <motion.div
                              layoutId="login-tab-active"
                              className="absolute inset-0 rounded-md bg-background shadow-sm"
                              transition={
                                prefersReduced
                                  ? { duration: 0 }
                                  : { type: "spring", stiffness: 400, damping: 32 }
                              }
                            />
                          )}
                          <span className="relative z-10">
                            {msg(
                              tab === "signin" ? "auth.login.tab_signin" : "auth.login.tab_signup",
                            )}
                          </span>
                        </button>
                      ))}
                    </div>

                    {(showOAuth || showPasskey) && (
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
                              <GithubLogo className="size-[18px]" />
                              {msg("auth.login.with_github")}
                            </Button>
                          )}
                          {showPasskey && (
                            <Button
                              type="button"
                              variant="outline"
                              size="lg"
                              disabled={passkeyLoading}
                              onClick={() => void handlePasskey()}
                              className="h-11 w-full gap-2.5 text-[0.9375rem] font-medium"
                            >
                              {passkeyLoading ? (
                                <CircleNotch className="size-[18px] animate-spin" />
                              ) : (
                                <Fingerprint className="size-[18px]" />
                              )}
                              {msg("auth.login.passkey")}
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
                          autoComplete="username webauthn"
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
                        <AnimatePresence initial={false}>
                          {passwordRules.map((rule) => (
                            <motion.p
                              key={rule}
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              transition={{ duration: 0.2 }}
                              className="mt-1.5 text-xs text-muted-foreground"
                            >
                              {msg(rule)}
                            </motion.p>
                          ))}
                        </AnimatePresence>
                      </div>

                      {authMode === "signin" && badCredentials && (
                        <button
                          type="button"
                          onClick={enterReset}
                          className="block w-full cursor-pointer text-end text-xs font-medium text-muted-foreground underline-offset-2 transition-colors hover:text-foreground hover:underline"
                        >
                          {msg("auth.login.forgot")}
                        </button>
                      )}

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
                        {loading && <CircleNotch className="size-4 animate-spin" />}
                        {msg(
                          authMode === "signin"
                            ? "auth.login.signin_submit"
                            : "auth.login.signup_submit",
                        )}
                      </Button>
                    </form>
                  </>
                )}
              </CardContent>
            </Card>

            <p className="mt-6 flex items-center justify-center gap-2 text-xs text-muted-foreground">
              <Link
                href={LEGAL_LINKS.terms}
                className="transition-colors hover:text-foreground hover:underline"
              >
                {msg("legal.terms_link")}
              </Link>
              <span aria-hidden="true" className="text-muted-foreground/40">
                {"·"}
              </span>
              <Link
                href={LEGAL_LINKS.privacy}
                className="transition-colors hover:text-foreground hover:underline"
              >
                {msg("legal.privacy_link")}
              </Link>
            </p>
          </div>
        )}
      </motion.div>
    </div>
  );
}
