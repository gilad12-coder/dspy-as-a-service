import type { Metadata, Viewport } from "next";
import { preload } from "react-dom";
import { cache } from "react";
import { cookies } from "next/headers";
import Script from "next/script";
import { AppShell } from "@/shared/layout/app-shell";
import { TooltipProvider } from "@/shared/ui/primitives/tooltip";
import {
  LocaleProvider,
  SessionProvider,
  ThemeProvider,
  ToastContainer,
} from "@/shared/providers";
import { SplashScreen } from "@/shared/layout/splash-screen";
import { TutorialOverlay, TutorialMenu, TutorialProvider } from "@/features/tutorial";
import {
  UserPrefsProvider,
  LiteModeProvider,
  SettingsModalProvider,
  SettingsModal,
} from "@/features/settings";
import { StorageQuotaModalHost } from "@/features/storage";
import { AppSkeletonTheme } from "@/shared/ui/skeleton";
import { msg } from "@/shared/lib/messages";
import { getServerRuntimeEnv, serializeRuntimeEnv } from "@/shared/lib/runtime-env";
import {
  DEFAULT_LOCALE,
  LOCALE_COOKIE,
  dirForLocale,
  isLocale,
  type Locale,
} from "@/shared/lib/locale";
import { serializeLocale, setServerLocale } from "@/shared/lib/runtime-locale";
import {
  serializeMessages,
  setServerMessages,
  type UiCatalog,
} from "@/shared/lib/runtime-messages";
import { buildActiveCatalog } from "@/shared/lib/messages.server";
import { getSiteUrl } from "@/shared/lib/site-config";
import "@fontsource-variable/heebo/index.css";
import "@fontsource-variable/inter/index.css";
import "@fontsource-variable/jetbrains-mono/index.css";
import "react-toastify/dist/ReactToastify.css";
import "./globals.css";

// The layout injects window.__SKYNET_ENV__ from getServerRuntimeEnv() so one
// built image can target any backend via the pod's runtime API_URL. That shim
// only works when the layout renders per-request: statically prerendered, it
// freezes the build-time default (localhost:8000, since API_URL isn't set in the
// Docker build) into every page and the browser then can't reach the backend.
// Force dynamic rendering so the injected env reflects the live pod env.
export const dynamic = "force-dynamic";

const siteUrl = getSiteUrl();
const siteName = "Skynet";

/** OpenGraph wants `language_TERRITORY`; the locale tag's hyphen maps to it. */
function ogLocale(locale: Locale): string {
  return locale.replace("-", "_");
}

/**
 * Resolve the request's locale: an explicit switcher cookie wins, else the
 * English default. New visitors land in English regardless of their browser's
 * `Accept-Language`; Hebrew is reached via the language switcher (which sets
 * the cookie). Wrapped in React `cache()` so the layout and `generateMetadata`
 * share a single cookie read per request.
 */
const resolveRequestLocale = cache(async (): Promise<Locale> => {
  const cookieLocale = (await cookies()).get(LOCALE_COOKIE)?.value;
  if (isLocale(cookieLocale)) return cookieLocale;
  return DEFAULT_LOCALE;
});

/**
 * Build the request's merged UI catalog once and share it across `generateMetadata`
 * and the layout via React `cache()`. This is the single payload the client ever
 * sees — the active locale's fallback chain, injected into the page rather than
 * bundled with every locale.
 */
const resolveActiveCatalog = cache(async (): Promise<UiCatalog> => {
  return buildActiveCatalog(await resolveRequestLocale());
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#FAF8F5" },
    { media: "(prefers-color-scheme: dark)", color: "#0F0C0A" },
  ],
};

export async function generateMetadata(): Promise<Metadata> {
  const locale = await resolveRequestLocale();
  setServerLocale(locale);
  setServerMessages(await resolveActiveCatalog());
  const siteDescription = msg("app.meta.description");
  return {
    title: {
      default: siteName,
      template: `%s | ${siteName}`,
    },
    description: siteDescription,
    icons: {
      icon: "/favicon.svg",
      apple: "/favicon.svg",
    },
    metadataBase: new URL(siteUrl),
    alternates: {
      canonical: "/",
    },
    openGraph: {
      type: "website",
      locale: ogLocale(locale),
      siteName,
      title: siteName,
      description: siteDescription,
      url: siteUrl,
    },
    twitter: {
      card: "summary_large_image",
      title: siteName,
      description: siteDescription,
    },
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        "max-video-preview": -1,
        "max-image-preview": "large",
        "max-snippet": -1,
      },
    },
  };
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const locale = await resolveRequestLocale();
  setServerLocale(locale);
  const messages = await resolveActiveCatalog();
  setServerMessages(messages);
  // Preload the above-the-fold variable subsets so the fallback→webfont swap
  // window (and its RTL line-box shift) is bounded. react-dom's preload()
  // dedupes to a single hoisted <link> per resource — a raw <link rel=preload> in
  // <head> gets emitted twice (literal + React's resource hoist). URLs are the
  // stable public/ copies the globals.css overrides point at; crossOrigin is
  // required since fonts fetch in CORS mode. JetBrains Mono stays non-preloaded.
  const fontPreload = { as: "font", type: "font/woff2", crossOrigin: "anonymous" } as const;
  preload("/fonts/heebo-hebrew-wght-normal.woff2", fontPreload);
  preload("/fonts/heebo-latin-wght-normal.woff2", fontPreload);
  preload("/fonts/inter-latin-wght-normal.woff2", fontPreload);
  const runtimeEnv = getServerRuntimeEnv();
  // dns-prefetch only helps when the API is on a different origin than the
  // document; on same-origin deploys the browser already resolved the host.
  let apiOrigin: string | null = null;
  try {
    const apiHost = new URL(runtimeEnv.apiUrl).origin;
    const siteHost = new URL(siteUrl).origin;
    if (apiHost !== siteHost) apiOrigin = apiHost;
  } catch {
    /* malformed URL — skip the hint */
  }
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    name: siteName,
    description: msg("app.meta.description"),
    url: siteUrl,
    applicationCategory: "DeveloperApplication",
    operatingSystem: "Web",
    inLanguage: locale,
  };
  // JSON.stringify does not escape `</` so a future field sourced from the
  // backend could prematurely close the surrounding <script>. Escape `<`
  // before injecting into dangerouslySetInnerHTML.
  const jsonLdSafe = JSON.stringify(jsonLd).replace(/</g, "\\u003c");
  return (
    <html lang={locale} dir={dirForLocale(locale)} suppressHydrationWarning>
      <head>
        <Script id="skynet-locale" strategy="beforeInteractive">
          {serializeLocale(locale)}
        </Script>
        <Script id="skynet-messages" strategy="beforeInteractive">
          {serializeMessages(messages)}
        </Script>
        <Script id="skynet-runtime-env" strategy="beforeInteractive">
          {serializeRuntimeEnv(runtimeEnv)}
        </Script>
        {apiOrigin && <link rel="dns-prefetch" href={apiOrigin} />}
        <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: jsonLdSafe }} />
      </head>
      <body suppressHydrationWarning>
        <LocaleProvider initialLocale={locale}>
          <SessionProvider>
            <UserPrefsProvider>
              <LiteModeProvider>
                <ThemeProvider>
                  <TooltipProvider>
                    <AppSkeletonTheme>
                      <SplashScreen />
                      <TutorialProvider>
                        <SettingsModalProvider>
                          <AppShell>{children}</AppShell>
                          <SettingsModal />
                        </SettingsModalProvider>
                        <TutorialOverlay />
                        <TutorialMenu />
                      </TutorialProvider>
                    </AppSkeletonTheme>
                  </TooltipProvider>
                </ThemeProvider>
              </LiteModeProvider>
            </UserPrefsProvider>
          </SessionProvider>
          <StorageQuotaModalHost />
          <ToastContainer />
        </LocaleProvider>
      </body>
    </html>
  );
}
