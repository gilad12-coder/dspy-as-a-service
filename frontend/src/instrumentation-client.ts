/** Initialize browser-side Sentry error monitoring when a public DSN is set. */

import * as Sentry from "@sentry/nextjs";

import { scrubSentryEvent, sentryTraceSampleRate } from "@/shared/lib/sentry-privacy";

const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;

Sentry.init({
  dsn,
  enabled: Boolean(dsn),
  environment: process.env.NEXT_PUBLIC_SENTRY_ENVIRONMENT ?? "development",
  release: process.env.NEXT_PUBLIC_APP_VERSION,
  sendDefaultPii: false,
  tracesSampleRate: sentryTraceSampleRate(process.env.NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE),
  // Replay only around errors, fully masked: it shows *what* the user did
  // before a crash without recording what they typed or looked at.
  replaysSessionSampleRate: 0,
  replaysOnErrorSampleRate: 1.0,
  integrations: [Sentry.replayIntegration({ maskAllText: true, blockAllMedia: true })],
  beforeSend: scrubSentryEvent,
});

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
