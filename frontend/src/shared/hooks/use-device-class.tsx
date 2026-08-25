"use client";

import * as React from "react";
import { PHONE_MEDIA_QUERY, type DeviceClass } from "@/shared/lib/device-class";

const DeviceClassContext = React.createContext<DeviceClass>("desktop");

/**
 * Publish the device class for the phone shell. `initial` is the server's
 * header-derived guess so the first paint already picks the right shell; after
 * mount the viewport (`matchMedia`) is authoritative and tracks resizes.
 */
export function DeviceClassProvider({
  initial,
  children,
}: {
  initial: DeviceClass;
  children: React.ReactNode;
}) {
  const [deviceClass, setDeviceClass] = React.useState<DeviceClass>(initial);

  React.useEffect(() => {
    const mq = window.matchMedia(PHONE_MEDIA_QUERY);
    const update = () => setDeviceClass(mq.matches ? "phone" : "desktop");
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  return <DeviceClassContext.Provider value={deviceClass}>{children}</DeviceClassContext.Provider>;
}

export function useDeviceClass(): DeviceClass {
  return React.useContext(DeviceClassContext);
}

/** True in the phone shell (viewport ≤767px, or the server hint before hydration). */
export function useIsPhone(): boolean {
  return useDeviceClass() === "phone";
}
