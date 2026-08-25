"use client";

import NextError from "next/error";
import { useEffect } from "react";
import { reportHandledError } from "@/shared/lib/report-error";

/** Capture otherwise-fatal App Router render errors and show a safe fallback. */
export default function GlobalError({ error }: { error: Error & { digest?: string } }) {
  useEffect(() => {
    reportHandledError(error, { tags: { source: "global_error", handled: false } });
  }, [error]);

  return (
    <html>
      <body>
        <NextError statusCode={0} />
      </body>
    </html>
  );
}
