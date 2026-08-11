import { NextResponse } from "next/server";

/**
 * Server-side proxy for "email me a password-reset code" (forgot-password step
 * one). The browser never holds the shared backend secret, so the login form
 * POSTs here; this handler forwards to the backend's internal
 * /auth/password-reset/request with the X-Internal-Auth secret attached. The
 * backend returns the same acknowledgement for known and unknown addresses, so
 * this route can't be used to enumerate registered emails. Failures surface the
 * backend's semantic ``code`` so the form can localize them.
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
    return NextResponse.json({ error: "accounts.invalid_email" }, { status: 400 });
  }
  const fields = (body ?? {}) as Record<string, unknown>;
  const email = typeof fields.email === "string" ? fields.email.trim().toLowerCase() : "";
  if (!email) {
    return NextResponse.json({ error: "accounts.invalid_email" }, { status: 400 });
  }
  let res: Response;
  try {
    res = await fetch(`${backendBaseUrl}/auth/password-reset/request`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Auth": backendAuthSecret },
      body: JSON.stringify({ email }),
    });
  } catch {
    return NextResponse.json({ error: "accounts.email_send_failed" }, { status: 502 });
  }
  if (res.ok) return NextResponse.json({ ok: true });
  let code = "accounts.email_send_failed";
  try {
    const data = (await res.json()) as { code?: unknown };
    if (typeof data.code === "string") code = data.code;
  } catch {
    // Non-JSON error body — keep the generic code.
  }
  return NextResponse.json({ error: code }, { status: res.status });
}
