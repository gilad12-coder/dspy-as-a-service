/** Tests for the phone-shell device class helpers. */

import assert from "node:assert/strict";
import test from "node:test";

import { deviceClassFromRequest, isDesktopOnlyPath, isPhoneSettingsTab } from "./device-class.ts";

test("isDesktopOnlyPath blocks authoring routes and their descendants", () => {
  for (const p of [
    "/submit",
    "/datasets",
    "/datasets/abc/edit",
    "/datasets/share/tok",
    "/tagger",
    "/tagger/123",
    "/storage",
  ]) {
    assert.equal(isDesktopOnlyPath(p), true, p);
  }
});

test("isDesktopOnlyPath lets view routes and lookalike prefixes through", () => {
  for (const p of [
    "/",
    "/explore",
    "/optimizations/1",
    "/share/tok",
    "/submitted",
    "/storages",
    "/login",
  ]) {
    assert.equal(isDesktopOnlyPath(p), false, p);
  }
});

test("deviceClassFromRequest prefers the client hint", () => {
  assert.equal(deviceClassFromRequest("Mozilla/5.0 (Windows NT 10.0)", "?1"), "phone");
  assert.equal(
    deviceClassFromRequest("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Mobile/15E148", "?0"),
    "desktop",
  );
});

test("deviceClassFromRequest falls back to the UA mobile token", () => {
  const iphone =
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1";
  const android =
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Mobile Safari/537.36";
  const androidTablet =
    "Mozilla/5.0 (Linux; Android 14; SM-X910) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36";
  const ipad =
    "Mozilla/5.0 (iPad; CPU OS 12_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 Mobile/15E148 Safari/604.1";
  const mac =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36";
  assert.equal(deviceClassFromRequest(iphone, null), "phone");
  assert.equal(deviceClassFromRequest(android, null), "phone");
  assert.equal(deviceClassFromRequest(androidTablet, null), "desktop");
  assert.equal(deviceClassFromRequest(ipad, null), "desktop");
  assert.equal(deviceClassFromRequest(mac, null), "desktop");
  assert.equal(deviceClassFromRequest(null, null), "desktop");
});

test("isPhoneSettingsTab keeps only the view-first settings tabs", () => {
  for (const t of ["account", "billing", "usage", "about"]) {
    assert.equal(isPhoneSettingsTab(t), true, t);
  }
  for (const t of [
    "wizard",
    "tagging",
    "agent",
    "security",
    "privacy",
    "providers",
    "api",
    "admin",
  ]) {
    assert.equal(isPhoneSettingsTab(t), false, t);
  }
});
