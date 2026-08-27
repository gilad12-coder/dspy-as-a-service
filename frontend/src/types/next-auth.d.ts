import "next-auth";
import "next-auth/jwt";

declare module "next-auth" {
  interface User {
    role?: string;
    groups?: string[];
    firstLogin?: boolean;
  }
  interface Session {
    backendAccessToken?: string;
    user: {
      name?: string | null;
      email?: string | null;
      image?: string | null;
      role?: string;
      groups?: string[];
      firstLogin?: boolean;
    };
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    role?: string;
    groups?: string[];
    firstLogin?: boolean;
  }
}
