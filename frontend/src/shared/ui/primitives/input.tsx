import * as React from "react";

import { cn } from "@/shared/lib/utils";

function Input({ className, type, dir, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      dir={dir as string | undefined}
      data-slot="input"
      className={cn(
        "h-9 w-full min-w-0 rounded-xl border border-input/90 bg-background/75 px-3 py-1 text-base text-start shadow-[inset_0_1px_0_rgba(255,255,255,0.72),0_12px_26px_-24px_rgba(15,23,42,0.45)] backdrop-blur-sm transition-[color,box-shadow,border-color,background-color] outline-none selection:bg-primary selection:text-primary-foreground file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-start placeholder:text-muted-foreground/90 disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 aria-invalid:border-destructive aria-invalid:ring-destructive/20",
        className,
      )}
      {...props}
    />
  );
}

export { Input };
