import { test } from "node:test";
import assert from "node:assert/strict";
import { proofBannerVariant } from "./proof-banner.ts";

test("no banner before the run settles", () => {
  assert.equal(proofBannerVariant(null, 5), null);
  assert.equal(proofBannerVariant(null, undefined), null);
});

test("refunded run reads as the guarantee holding", () => {
  assert.equal(proofBannerVariant({ outcome: "refunded", credits: 12 }, undefined), "refunded");
  assert.equal(proofBannerVariant({ outcome: "refunded", credits: 12 }, -3), "refunded");
});

test("billed run with real lift claims the beat", () => {
  assert.equal(proofBannerVariant({ outcome: "billed", credits: 12 }, 3), "billed-lift");
});

test("billed run with a regression never claims a beat", () => {
  // A re-run on a spent guarantee slot gets stamped billed even though the
  // optimizer regressed; the banner must fall back to neutral receipt copy.
  assert.equal(proofBannerVariant({ outcome: "billed", credits: 12 }, -3), "billed-neutral");
});

test("billed run with zero or unscored improvement falls back to neutral", () => {
  assert.equal(proofBannerVariant({ outcome: "billed", credits: 12 }, 0), "billed-neutral");
  assert.equal(proofBannerVariant({ outcome: "billed", credits: 12 }, undefined), "billed-neutral");
});
