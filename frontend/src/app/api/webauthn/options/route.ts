import { NextResponse } from "next/server";

/**
 * Server-side proxy for passkey sign-in options. The login page hits this
 * before any session exists; the handler forwards to the backend's internal
 * /auth/webauthn/options with the X-Internal-Auth secret attached (the
 * browser never holds it) and returns the WebAuthn request options that
 * navigator.credentials.get() consumes. The matching assertion is verified
 * inside the NextAuth "passkey" provider, not through a public route.
 */

export const runtime = "nodejs";

const backendBaseUrl =
  process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const backendAuthSecret = process.env.BACKEND_AUTH_SECRET ?? process.env.AUTH_SECRET;

export async function POST(): Promise<NextResponse> {
  if (!backendAuthSecret) {
    return NextResponse.json({ error: "auth.not_configured" }, { status: 500 });
  }
  let res: Response;
  try {
    res = await fetch(`${backendBaseUrl}/auth/webauthn/options`, {
      method: "POST",
      headers: { "X-Internal-Auth": backendAuthSecret },
    });
  } catch {
    return NextResponse.json({ error: "webauthn.invalid_credential" }, { status: 502 });
  }
  if (!res.ok) {
    return NextResponse.json({ error: "webauthn.invalid_credential" }, { status: res.status });
  }
  return NextResponse.json(await res.json());
}
