import { NextResponse } from "next/server";

/**
 * Server-side proxy for "set a new password with a reset code" (forgot-password
 * step two). Forwards to the backend's internal /auth/password-reset/confirm
 * with the X-Internal-Auth secret attached; the new password goes to the backend
 * and is never echoed back to the client. Failures surface the backend's
 * semantic ``code`` (e.g. ``accounts.invalid_reset_code``) so the form can
 * localize them.
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
    return NextResponse.json({ error: "accounts.invalid_reset_code" }, { status: 400 });
  }
  const fields = (body ?? {}) as Record<string, unknown>;
  const email = typeof fields.email === "string" ? fields.email.trim().toLowerCase() : "";
  const code = typeof fields.code === "string" ? fields.code.trim() : "";
  const newPassword = typeof fields.new_password === "string" ? fields.new_password : "";
  if (!email || !code || !newPassword) {
    return NextResponse.json({ error: "accounts.invalid_reset_code" }, { status: 400 });
  }
  let res: Response;
  try {
    res = await fetch(`${backendBaseUrl}/auth/password-reset/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Auth": backendAuthSecret },
      body: JSON.stringify({ email, code, new_password: newPassword }),
    });
  } catch {
    return NextResponse.json({ error: "auth.login.reset_failed" }, { status: 502 });
  }
  if (res.ok) return NextResponse.json({ ok: true });
  let errorCode = "auth.login.reset_failed";
  try {
    const data = (await res.json()) as { code?: unknown };
    if (typeof data.code === "string") errorCode = data.code;
  } catch {
    // Non-JSON error body — keep the generic code.
  }
  return NextResponse.json({ error: errorCode }, { status: res.status });
}
