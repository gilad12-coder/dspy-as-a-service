import { test } from "node:test";
import assert from "node:assert/strict";
import type { OptimizationStatusResponse } from "@/shared/types/api";
import { canCompareKeys, compareCompatibilityKey } from "./compatibility.ts";

const job = (o: Partial<OptimizationStatusResponse>): OptimizationStatusResponse =>
  o as OptimizationStatusResponse;

test("canCompareKeys needs at least two rows", () => {
  assert.equal(canCompareKeys([]), false);
  assert.equal(canCompareKeys(["a"]), false);
});

test("canCompareKeys requires every key present and identical", () => {
  assert.equal(canCompareKeys(["a", "a"]), true);
  assert.equal(canCompareKeys(["a", "a", "a"]), true);
  assert.equal(canCompareKeys(["a", "b"]), false);
  assert.equal(canCompareKeys([null, "a"]), false);
  assert.equal(canCompareKeys(["a", null]), false);
});

test("compareCompatibilityKey returns the fingerprint only when it is a non-empty string", () => {
  assert.equal(compareCompatibilityKey(job({ compare_fingerprint: "fp-1" })), "fp-1");
  assert.equal(compareCompatibilityKey(job({ compare_fingerprint: "" })), null);
  assert.equal(compareCompatibilityKey(job({ compare_fingerprint: null })), null);
  assert.equal(compareCompatibilityKey(job({})), null);
});
