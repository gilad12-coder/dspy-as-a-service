"use client";

import * as React from "react";
import { Popover as PopoverPrimitive } from "radix-ui";

import { cn } from "@/shared/lib/utils";

function Popover({ ...props }: React.ComponentProps<typeof PopoverPrimitive.Root>) {
  return <PopoverPrimitive.Root data-slot="popover" {...props} />;
}

function PopoverTrigger({ ...props }: React.ComponentProps<typeof PopoverPrimitive.Trigger>) {
  return <PopoverPrimitive.Trigger data-slot="popover-trigger" {...props} />;
}

function PopoverContent({
  className,
  sideOffset = 6,
  align = "center",
  children,
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Content>) {
  return (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Content
        data-slot="popover-content"
        sideOffset={sideOffset}
        align={align}
        className={cn(
          "origin-(--radix-popover-content-transform-origin)",
          // Unfold from the trigger origin (200ms expo-out) instead of the
          // default ~150ms snap, with a crisper 140ms ease-in dismiss. The
          // arbitrary animation-* props set duration/easing on the enter/exit
          // keyframes directly — the plugin's `duration-*` utility targets
          // transition-duration, not these keyframes.
          "animate-in fade-in-0 zoom-in-95",
          "[animation-duration:200ms] [animation-timing-function:cubic-bezier(0.16,1,0.3,1)]",
          "data-[side=bottom]:slide-in-from-top-2",
          "data-[side=left]:slide-in-from-right-2",
          "data-[side=right]:slide-in-from-left-2",
          "data-[side=top]:slide-in-from-bottom-2",
          "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
          "data-[state=closed]:[animation-duration:140ms] data-[state=closed]:[animation-timing-function:cubic-bezier(0.4,0,1,1)]",
          "motion-reduce:animate-none",
          "rounded-xl border border-border/60 bg-background/95 backdrop-blur-xl shadow-lg",
          "outline-none",
          className,
        )}
        style={{ zIndex: 50 }}
        {...props}
      >
        {children}
      </PopoverPrimitive.Content>
    </PopoverPrimitive.Portal>
  );
}

export { Popover, PopoverTrigger, PopoverContent };
