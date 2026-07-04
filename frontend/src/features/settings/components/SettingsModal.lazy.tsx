"use client";

import dynamic from "next/dynamic";

// The settings modal (1300+ lines, Recharts via UsageTab, @lobehub/icons via
// ByokKeysSection) never renders on first paint — the dialog starts closed —
// yet the always-mounted root layout pulls its entire static import graph
// into the shared first-load chunk on every route. Re-export it lazily so the
// chunk fetches post-hydration, off the first-paint critical path. Kept in a
// dedicated "use client" module because `ssr: false` is only valid in Client
// Components.
export const SettingsModal = dynamic(() => import("./SettingsModal").then((m) => m.SettingsModal), {
  ssr: false,
});
