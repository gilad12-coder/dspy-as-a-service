/** Guard tutorial routes, targets, tracks, and localized copy against product drift. */

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const STEPS_PATH = join(HERE, "steps.ts");
const MENU_PATH = join(HERE, "../components/tutorial-menu.tsx");
const SRC_PATH = fileURLToPath(new URL("../../../", import.meta.url));
const EN_PATH = fileURLToPath(new URL("../../../../../i18n/locales/ui/en.json", import.meta.url));
const HE_PATH = fileURLToPath(new URL("../../../../../i18n/locales/ui/he.json", import.meta.url));

function readSourceTree(directory: string): string {
  return readdirSync(directory, { withFileTypes: true })
    .flatMap((entry) => {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) return readSourceTree(path);
      if (![".ts", ".tsx"].includes(extname(entry.name)) || path === STEPS_PATH) return [];
      return [readFileSync(path, "utf8")];
    })
    .join("\n");
}

test("every tutorial spotlight target is still declared by the application", () => {
  const steps = readFileSync(STEPS_PATH, "utf8");
  const appSource = readSourceTree(SRC_PATH);
  const targets = [...steps.matchAll(/target: "\[data-tutorial='([^']+)'\]"/g)].map(
    (match) => match[1],
  );

  assert.ok(targets.length > 0);
  for (const target of targets) {
    assert.ok(
      appSource.includes(`"${target}"`) || appSource.includes(`'${target}'`),
      `Missing application target for tutorial step: ${target}`,
    );
  }
});

test("tutorial workflow tracks stay synchronized with the chooser", () => {
  const steps = readFileSync(STEPS_PATH, "utf8");
  const menu = readFileSync(MENU_PATH, "utf8");
  const tracks = ["quick", "data", "build", "results", "workspace"];

  for (const track of tracks) {
    assert.match(steps, new RegExp(`\\b${track}: \\{`));
    assert.ok(menu.includes(`id: "${track}"`));
  }
  assert.doesNotMatch(steps, /deep-dive/);
  assert.doesNotMatch(menu, /deep-dive/);
});

test("tutorial-owned message keys exist in both base catalogs", () => {
  const steps = readFileSync(STEPS_PATH, "utf8");
  const menu = readFileSync(MENU_PATH, "utf8");
  const en = JSON.parse(readFileSync(EN_PATH, "utf8")) as Record<string, string>;
  const he = JSON.parse(readFileSync(HE_PATH, "utf8")) as Record<string, string>;
  const keys = new Set(
    [...`${steps}\n${menu}`.matchAll(/"(tutorial\.[^"]+)"/g)].map((match) => match[1]),
  );

  for (const key of keys) {
    assert.ok(key in en, `Missing English tutorial message: ${key}`);
    assert.ok(key in he, `Missing Hebrew tutorial message: ${key}`);
  }
});
