import { test } from "node:test";
import assert from "node:assert/strict";
import { readBilling } from "./run-billing.ts";

test("no stamp before the run settles", () => {
  assert.equal(readBilling(undefined), null);
  assert.equal(readBilling({}), null);
});

test("a billed stamp is returned intact", () => {
  const billing = { outcome: "billed", credits: 12, estimated_low: 8, estimated_high: 20 };
  assert.deepEqual(readBilling({ billing }), billing);
});

test("malformed stamps are rejected", () => {
  assert.equal(readBilling({ billing: "billed" }), null);
  assert.equal(readBilling({ billing: { outcome: "billed" } }), null);
});

test("a legacy refunded stamp is ignored, not rendered", () => {
  // Stamped by the retired no-lift guarantee; its credits describe a refund,
  // not a charge, so surfacing it as a cost would misread history.
  assert.equal(readBilling({ billing: { outcome: "refunded", credits: 12 } }), null);
});
