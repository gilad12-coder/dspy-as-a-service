import { NextResponse } from "next/server";

/**
 * Server-side proxy for "email me a sign-in code". Hit from the login form's
 * 2FA step (no session yet); forwards to the backend's internal
 * /auth/2fa/email/send with the X-Internal-Auth secret attached. The backend
 * re-verifies the password before sending anything, so this route can't be
 * used to spam arbitrary inboxes. Failures surface the backend's semantic
 * ``code`` so the form can localize them.
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
    return NextResponse.json({ error: "accounts.invalid_credentials" }, { status: 400 });
  }
  const fields = (body ?? {}) as Record<string, unknown>;
  const email = typeof fields.email === "string" ? fields.email.trim().toLowerCase() : "";
  const password = typeof fields.password === "string" ? fields.password : "";
  if (!email || !password) {
    return NextResponse.json({ error: "accounts.invalid_credentials" }, { status: 400 });
  }
  let res: Response;
  try {
    res = await fetch(`${backendBaseUrl}/auth/2fa/email/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Auth": backendAuthSecret },
      body: JSON.stringify({ email, password }),
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
