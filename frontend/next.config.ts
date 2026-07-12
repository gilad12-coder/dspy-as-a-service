import type { NextConfig } from "next";

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
      "lucide-react",
      "radix-ui",
      "@radix-ui/react-direction",
      "class-variance-authority",
      "clsx",
      "tailwind-merge",
      "recharts",
      "@fontsource-variable/heebo",
      "@fontsource-variable/geist",
      "@fontsource-variable/jetbrains-mono",
      "@uiw/react-codemirror",
      "@codemirror/lang-python",
      "xlsx",
      "@lobehub/icons",
    ],
  },
};

export default nextConfig;
