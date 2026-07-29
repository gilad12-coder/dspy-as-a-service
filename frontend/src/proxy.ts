import { auth } from "@/shared/lib/auth";

export default auth;

export const config = {
  matcher: [
    // Every app surface requires login — including ``/share/<token>``. A
    // recipient must authenticate so the backend resolves them to the role the
    // link grants (e.g. editor); a logged-out visitor bounces to /login and
    // returns via callbackUrl. ``api/auth`` (NextAuth), ``api/register``,
    // ``api/webauthn`` (passkey sign-in options), and ``api/2fa`` (emailed
    // sign-in codes) are excluded: all are hit by logged-out visitors, so
    // guarding them would bounce the POSTs to /login and the flow would never
    // complete.
    "/((?!login|api/auth|api/register|api/webauthn|api/2fa|_next/static|_next/image|favicon\\.svg|robots\\.txt|sitemap\\.xml).*)",
  ],
};
