"use client";

import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { track, TelemetryEvent } from "@/shared/lib/telemetry";

/**
 * Headless telemetry instrumentation: renders nothing, just wires autocapture.
 *
 * Two automatic streams: a `page_view` on first mount and every route change,
 * and a `element_click` per meaningful click (delegated at the document, so it
 * needs no wrapping). It carries no children and reads no React context, so it
 * mounts as a single self-closing sibling anywhere under the app — attribution
 * to a signed-in user is handled by the API layer's bearer token, not here.
 */
export function TelemetryProvider() {
  const pathname = usePathname();
  const lastPath = useRef<string | null>(null);

  // Query strings are intentionally excluded from the path — they can carry
  // tokens or search terms (PII). The ref dedupes React's double-invoke and
  // any same-path re-render.
  useEffect(() => {
    if (pathname === null || pathname === lastPath.current) return;
    lastPath.current = pathname;
    track(TelemetryEvent.PageView, { path: pathname });
  }, [pathname]);

  useEffect(() => {
    function onClick(event: MouseEvent) {
      const descriptor = describeTarget(event.target as Element | null);
      if (descriptor) track(TelemetryEvent.ElementClick, descriptor);
    }
    document.addEventListener("click", onClick, { capture: true });
    return () => document.removeEventListener("click", onClick, { capture: true });
  }, []);

  return null;
}

const INTERACTIVE_TAGS = new Set(["BUTTON", "A"]);
const MAX_VALUE_LEN = 64;

/**
 * Resolve a click target to a privacy-safe descriptor, or null to ignore it.
 *
 * Climbs from the clicked node to the nearest element that either carries a
 * `data-telemetry` marker (an explicit, labelled event) or is itself an
 * interactive control. A plain non-interactive click resolves to null and is
 * dropped, so the stream stays signal over noise.
 */
function describeTarget(start: Element | null): Record<string, unknown> | null {
  let el: Element | null = start;
  for (let depth = 0; el && depth < 8; depth++) {
    const explicit = el.getAttribute?.("data-telemetry");
    if (explicit) return { label: clip(explicit), ...dataTelemetryExtras(el) };
    const role = el.getAttribute?.("role");
    const interactive =
      INTERACTIVE_TAGS.has(el.tagName) || role === "button" || el.hasAttribute?.("data-testid");
    if (interactive) return describeElement(el);
    el = el.parentElement;
  }
  return null;
}

/** Pull `data-telemetry-*` attributes into a props object (clipped values). */
function dataTelemetryExtras(el: Element): Record<string, unknown> {
  const extras: Record<string, unknown> = {};
  for (const attr of Array.from(el.attributes)) {
    if (attr.name.startsWith("data-telemetry-")) {
      const key = attr.name.slice("data-telemetry-".length);
      if (key) extras[key] = clip(attr.value);
    }
  }
  return extras;
}

/**
 * Structural descriptor for a generic interactive element. Deliberately records
 * only stable identifiers (tag, role, id, test id, name, type, link path) and
 * never the element's text or any input value — those can contain user content.
 */
function describeElement(el: Element): Record<string, unknown> {
  const out: Record<string, unknown> = { tag: el.tagName.toLowerCase() };
  const role = el.getAttribute("role");
  if (role) out.role = role;
  if (el.id) out.id = clip(el.id);
  const testid = el.getAttribute("data-testid");
  if (testid) out.testid = clip(testid);
  const name = el.getAttribute("name");
  if (name) out.name = clip(name);
  const type = el.getAttribute("type");
  if (type) out.type = clip(type);
  if (el.tagName === "A") {
    const href = el.getAttribute("href");
    if (href) out.href = hrefPath(href);
  }
  return out;
}

/** Truncate a string to the value cap so descriptors stay small. */
function clip(value: string, max = MAX_VALUE_LEN): string {
  return value.length > max ? value.slice(0, max) : value;
}

/** Reduce an href to its path (drop query/hash; keep cross-origin host). */
function hrefPath(href: string): string {
  try {
    const url = new URL(href, window.location.origin);
    return url.origin === window.location.origin
      ? url.pathname
      : `${url.origin}${url.pathname}`;
  } catch {
    return clip(href);
  }
}
