import { test } from "node:test";
import assert from "node:assert/strict";

import { isReactModuleName } from "./is-react-module.ts";

test("recognizes ReAct module names regardless of case or surrounding whitespace", () => {
  assert.equal(isReactModuleName("react"), true);
  assert.equal(isReactModuleName(" ReAct "), true);
});

test("does not route other or missing modules to the ReAct playground", () => {
  assert.equal(isReactModuleName("predict"), false);
  assert.equal(isReactModuleName(null), false);
  assert.equal(isReactModuleName(undefined), false);
});
