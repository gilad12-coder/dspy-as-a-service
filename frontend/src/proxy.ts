import { auth } from "@/shared/lib/auth";

export default auth;

export const config = {
  matcher: [
    // NextAuth and static assets must stay reachable before login. Every actual
    // product route, including shared links and optimization details, requires
    // an authenticated local or ADFS identity.
    "/((?!login|api/auth|_next/static|_next/image|favicon\\.svg|robots\\.txt).*)",
  ],
};
