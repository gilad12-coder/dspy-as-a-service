import { NextResponse } from "next/server";

/**
 * Server-side proxy for "create an account in Skynet". The browser never holds
 * the shared backend secret, so signup POSTs here; this handler forwards to the
 * backend's internal /auth/register with the X-Internal-Auth secret attached.
 * On failure it surfaces the backend's semantic i18n ``code`` (e.g.
 * ``accounts.email_taken``) so the login form can localize the message; it never
 * forwards the password back to the client.
 */

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
  const password = typeof fields.password === "string" ? fields.password : "";
  const name = typeof fields.name === "string" ? fields.name.trim() : "";
  if (!email || !password) {
    return NextResponse.json({ error: "accounts.invalid_email" }, { status: 400 });
  }

  let res: Response;
  try {
    res = await fetch(`${backendBaseUrl}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Auth": backendAuthSecret },
      body: JSON.stringify({ email, password, name }),
    });
  } catch {
    return NextResponse.json({ error: "auth.login.register_failed" }, { status: 502 });
  }

  if (res.ok) return NextResponse.json({ ok: true }, { status: 201 });

  let code = "auth.login.register_failed";
  try {
    const data = (await res.json()) as { code?: unknown };
    if (typeof data.code === "string") code = data.code;
  } catch {
    // Non-JSON error body — keep the generic code.
  }
  return NextResponse.json({ error: code }, { status: res.status });
}
