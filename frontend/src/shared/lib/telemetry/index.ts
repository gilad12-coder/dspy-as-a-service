/**
 * Public surface of the first-party telemetry SDK.
 *
 * App code imports from here: `track` for events and `TelemetryEvent` for the
 * canonical names. The opt-out controls live in `./client`; the autocapture
 * wiring lives in `shared/providers/telemetry-provider`.
 */

export { track } from "./client";
export { TelemetryEvent } from "./events";
