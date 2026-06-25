import { auth } from "@/shared/lib/auth";

export default auth;

export const config = {
  matcher: [
    // Every app surface requires login — including ``/share/<token>``. A
    // recipient must authenticate so the backend resolves them to the role the
    // link grants (e.g. editor); a logged-out visitor bounces to /login and
    // returns via callbackUrl. ``api/auth`` (NextAuth) and ``api/register`` are
    // excluded: both are hit by logged-out visitors, so guarding them would
    // bounce sign-up POSTs to /login and the account would never be created.
    "/((?!login|api/auth|api/register|_next/static|_next/image|favicon\\.svg|robots\\.txt|sitemap\\.xml).*)",
  ],
};
