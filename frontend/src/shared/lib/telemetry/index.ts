/**
 * Public surface of the first-party telemetry SDK.
 *
 * App code imports from here: `track` for events, `TelemetryEvent` for the
 * canonical names, and the opt-out controls. The autocapture wiring lives in
 * `shared/providers/telemetry-provider`.
 */

export { track, flush, setTelemetryOptOut, isTelemetryOptedOut } from "./client";
export { TelemetryEvent, type TelemetryEventName } from "./events";
export { getAnonymousId, getSessionId } from "./ids";
