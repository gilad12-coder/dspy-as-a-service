import { NextResponse } from "next/server";

/**
 * Server-side proxy for "confirm my email with the code you emailed" (the verify
 * step on the login screen). Forwards to the backend's internal
 * /auth/email-verify/confirm with the X-Internal-Auth secret attached.
 * Confirming grants no session — it only flips the account's verified flag, so
 * the user still signs in afterwards with their password. Failures surface the
 * backend's semantic ``code`` (e.g. ``accounts.invalid_verification_code``) so
 * the form can localize them.
 */

export const runtime = "nodejs";

const backendBaseUrl =
  process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const backendAuthSecret = process.env.BACKEND_AUTH_SECRET ?? process.env.AUTH_SECRET;

export async function POST(request: Request): Promise<NextResponse> {
  if (!backendAuthSecret) {
    return NextResponse.json({ error: "auth.not_configured" }, { status: 500 });
  }
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "accounts.invalid_verification_code" }, { status: 400 });
  }
  const fields = (body ?? {}) as Record<string, unknown>;
  const email = typeof fields.email === "string" ? fields.email.trim().toLowerCase() : "";
  const code = typeof fields.code === "string" ? fields.code.trim() : "";
  if (!email || !code) {
    return NextResponse.json({ error: "accounts.invalid_verification_code" }, { status: 400 });
  }
  let res: Response;
  try {
    res = await fetch(`${backendBaseUrl}/auth/email-verify/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Auth": backendAuthSecret },
      body: JSON.stringify({ email, code }),
    });
  } catch {
    return NextResponse.json({ error: "auth.login.verify_failed" }, { status: 502 });
  }
  if (res.ok) return NextResponse.json({ ok: true });
  let errorCode = "auth.login.verify_failed";
  try {
    const data = (await res.json()) as { code?: unknown };
    if (typeof data.code === "string") errorCode = data.code;
  } catch {
    // Non-JSON error body — keep the generic code.
  }
  return NextResponse.json({ error: errorCode }, { status: res.status });
}
