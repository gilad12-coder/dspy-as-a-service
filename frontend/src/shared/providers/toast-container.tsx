"use client";

import { ToastContainer as ReactToastContainer } from "react-toastify";
import { getActiveDir } from "@/shared/lib/runtime-locale";

export function ToastContainer() {
  // Follow the active locale: Hebrew toasts read RTL and anchor at the inline
  // edge (bottom-left); English toasts read LTR and mirror to bottom-right, away
  // from the now-left-hand sidebar.
  const isRtl = getActiveDir() === "rtl";
  return (
    <ReactToastContainer
      position={isRtl ? "bottom-left" : "bottom-right"}
      autoClose={4000}
      hideProgressBar={false}
      newestOnTop
      closeOnClick
      closeButton={false}
      rtl={isRtl}
      pauseOnFocusLoss={false}
      draggable
      pauseOnHover
      theme="light"
      toastClassName="text-sm !py-3 !px-4"
    />
  );
}
