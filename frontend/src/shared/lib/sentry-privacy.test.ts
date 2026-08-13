/** Tests for Sentry privacy filters and trace sampling. */

import assert from "node:assert/strict";
import test from "node:test";

import { scrubSentryEvent, sentryTraceSampleRate } from "./sentry-privacy.ts";

test("scrubSentryEvent removes identity, secrets, and URL queries", () => {
  const event = {
    user: { email: "alice@example.com" },
    request: {
      url: "https://skynetml.com/submit?token=secret#fragment",
      headers: { authorization: "secret" },
      cookies: { session: "secret" },
      data: { prompt: "private" },
    },
  };

  assert.deepEqual(scrubSentryEvent(event), {
    user: undefined,
    request: {
      url: "https://skynetml.com/submit",
      headers: undefined,
      cookies: undefined,
      data: undefined,
    },
  });
});

test("sentryTraceSampleRate accepts bounded values and defaults invalid input", () => {
  assert.equal(sentryTraceSampleRate("0"), 0);
  assert.equal(sentryTraceSampleRate("0.25"), 0.25);
  assert.equal(sentryTraceSampleRate("1"), 1);
  assert.equal(sentryTraceSampleRate("2"), 0.05);
  assert.equal(sentryTraceSampleRate("nope"), 0.05);
});
