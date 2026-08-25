import assert from "node:assert/strict";
import test from "node:test";

import { prefillFreetextPredictions } from "./freetext-prefill.ts";

test("prefills generated free-text labels for a restored review round", () => {
  const annotations = prefillFreetextPredictions(
    {},
    {
      "1": { value: "82", confidence: 1 },
      "4": { value: "11", confidence: 1 },
    },
    ["1", "4"],
  );

  assert.deepEqual(annotations, { "1": "82", "4": "11" });
});

test("does not replace a human edit or insert an empty prediction", () => {
  const original = { "1": "eighty-two" };
  const annotations = prefillFreetextPredictions(
    original,
    {
      "1": { value: "82", confidence: 1 },
      "4": { value: "", confidence: 0 },
    },
    ["1", "4"],
  );

  assert.equal(annotations, original);
});
