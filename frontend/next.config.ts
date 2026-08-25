import type { NextConfig } from "next";
import { withSentryConfig } from "@sentry/nextjs";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  compress: true,
  poweredByHeader: false,
  images: {
    formats: ["image/avif", "image/webp"],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920],
    imageSizes: [16, 32, 48, 64, 96, 128, 256],
  },
  headers: async () => [
    {
      source: "/(.*)",
      headers: [
        { key: "X-Content-Type-Options", value: "nosniff" },
        { key: "X-Frame-Options", value: "DENY" },
        { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        // HSTS: after the first HTTPS response, browsers refuse plain HTTP for
        // two years. All skynetml.com hosts (apex + www) terminate TLS on
        // Railway; the API runs on a separate *.up.railway.app host, so
        // includeSubDomains stays safe. No `preload` — that opts into the
        // browser preload list, which is slow to reverse.
        {
          key: "Strict-Transport-Security",
          value: "max-age=63072000; includeSubDomains",
        },
      ],
    },
    {
      source: "/favicon.svg",
      headers: [{ key: "Cache-Control", value: "public, max-age=31536000, immutable" }],
    },
    {
      source: "/robots.txt",
      headers: [{ key: "Cache-Control", value: "public, max-age=86400" }],
    },
    // Next.js already serves content-hashed /_next/static assets with
    // `immutable` long-lived caching; setting our own Cache-Control there is
    // redundant and trips a build-time warning, so it's intentionally omitted.
  ],
  experimental: {
    optimizePackageImports: [
      "framer-motion",
      "react-toastify",
      "radix-ui",
      "@radix-ui/react-direction",
      "class-variance-authority",
      "clsx",
      "tailwind-merge",
      "recharts",
      "@fontsource-variable/heebo",
      "@fontsource-variable/geist",
      "@fontsource-variable/jetbrains-mono",
      "@fontsource-variable/noto-sans-arabic",
      "@fontsource-variable/vazirmatn",
      "@fontsource-variable/noto-sans-devanagari",
      "@fontsource-variable/noto-sans-jp",
      "@fontsource-variable/noto-sans-kr",
      "@fontsource-variable/noto-sans-sc",
      "@uiw/react-codemirror",
      "@codemirror/lang-python",
      "xlsx",
      "@lobehub/icons",
    ],
  },
};

export default withSentryConfig(nextConfig, {
  org: process.env.SENTRY_ORG,
  project: process.env.SENTRY_PROJECT,
  authToken: process.env.SENTRY_AUTH_TOKEN,
  silent: true,
  webpack: { treeshake: { removeDebugLogging: true } },
  telemetry: false,
});
