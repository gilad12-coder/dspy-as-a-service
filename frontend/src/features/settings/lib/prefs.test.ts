import { test } from "node:test";
import assert from "node:assert/strict";
import { parseAgentPreferencePatch } from "./prefs.ts";

test("parseAgentPreferencePatch accepts the validated agent response envelope", () => {
  assert.deepEqual(
    parseAgentPreferencePatch({
      updates: {
        advanced_mode: true,
        wizard_split_mode: "manual",
        agent_trust_mode: "yolo",
      },
    }),
    {
      advancedMode: true,
      wizardSplitMode: "manual",
    },
  );
});

test("parseAgentPreferencePatch handles nested JSON and ignores invalid fields", () => {
  assert.deepEqual(
    parseAgentPreferencePatch(
      JSON.stringify({
        result: {
          updates: {
            tagger_assist: false,
            wizard_code_assist: "unknown",
            unsupported: true,
          },
        },
      }),
    ),
    { taggerAssist: false },
  );
});
