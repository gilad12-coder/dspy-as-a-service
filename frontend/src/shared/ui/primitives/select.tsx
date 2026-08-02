"use client";

import * as React from "react";
import { CaretDown, CaretUp, Check } from "@/shared/ui/icons";
import { Select as SelectPrimitive } from "radix-ui";

import { cn } from "@/shared/lib/utils";

function Select({ ...props }: React.ComponentProps<typeof SelectPrimitive.Root>) {
  return <SelectPrimitive.Root data-slot="select" {...props} />;
}

function SelectValue({ ...props }: React.ComponentProps<typeof SelectPrimitive.Value>) {
  return <SelectPrimitive.Value data-slot="select-value" {...props} />;
}

function SelectTrigger({
  className,
  size = "default",
  children,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.Trigger> & {
  size?: "sm" | "default";
}) {
  return (
    <SelectPrimitive.Trigger
      data-slot="select-trigger"
      data-size={size}
      className={cn(
        "group/select-trigger flex w-fit select-none items-center justify-between gap-2 rounded-md border border-input bg-transparent px-3 py-2 text-sm whitespace-nowrap shadow-xs cursor-pointer transform-gpu transition-[transform,background-color,color,border-color,box-shadow] duration-120 ease-[cubic-bezier(0.2,0.8,0.2,1)] outline-none hover:border-foreground/20 hover:bg-accent/55 hover:shadow-sm data-[state=open]:border-foreground/25 data-[state=open]:bg-accent/60 data-[state=open]:shadow-sm focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 motion-reduce:transform-none motion-reduce:transition-none aria-invalid:border-destructive aria-invalid:ring-destructive/20 data-[placeholder]:text-muted-foreground data-[size=default]:h-9 data-[size=sm]:h-8 *:data-[slot=select-value]:line-clamp-1 *:data-[slot=select-value]:flex *:data-[slot=select-value]:items-center *:data-[slot=select-value]:gap-2 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 [&_svg:not([class*='text-'])]:text-muted-foreground",
        className,
      )}
      {...props}
    >
      {children}
      <SelectPrimitive.Icon asChild>
        <CaretDown className="size-4 opacity-50 transition-[transform,opacity] duration-120 ease-[cubic-bezier(0.2,0.8,0.2,1)] group-hover/select-trigger:opacity-80 group-data-[state=open]/select-trigger:rotate-180 group-data-[state=open]/select-trigger:opacity-80" />
      </SelectPrimitive.Icon>
    </SelectPrimitive.Trigger>
  );
}

function SelectContent({
  className,
  children,
  // "popper" anchors the menu to the trigger (drops below/above it). The Radix
  // default "item-aligned" scrolls the page to align the selected option over
  // the trigger, which visibly jumps the surrounding layout when the trigger
  // sits low on a scrollable page. Popper avoids that scroll entirely.
  position = "popper",
  align = "center",
  sideOffset = 4,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.Content>) {
  return (
    <SelectPrimitive.Portal>
      <SelectPrimitive.Content
        data-slot="select-content"
        className={cn(
          "scrollbar-subtle relative z-50 max-h-(--radix-select-content-available-height) min-w-[8rem] origin-(--radix-select-content-transform-origin) overflow-x-hidden overflow-y-auto rounded-xl border border-border/70 bg-popover/95 text-popover-foreground shadow-lg backdrop-blur-sm data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95",
          position === "popper" &&
            "data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1",
          className,
        )}
        style={{ zIndex: 9999 }}
        position={position}
        align={align}
        sideOffset={sideOffset}
        {...props}
      >
        <SelectScrollUpButton />
        <SelectPrimitive.Viewport
          className={cn(
            "p-1",
            position === "popper" &&
              "w-full min-w-[var(--radix-select-trigger-width)] scroll-my-1",
          )}
        >
          {children}
        </SelectPrimitive.Viewport>
        <SelectScrollDownButton />
      </SelectPrimitive.Content>
    </SelectPrimitive.Portal>
  );
}

function SelectItem({
  className,
  children,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.Item>) {
  // Highlight (pointer hover + keyboard) is driven entirely by Radix's
  // data-highlighted state in CSS — no imperative style mutation — so pointer
  // and keyboard navigation share one consistent, flicker-free appearance.
  // The selected row gets a clear check indicator, not just a tint.
  return (
    <SelectPrimitive.Item
      data-slot="select-item"
      className={cn(
        "relative flex w-full cursor-pointer items-center justify-between gap-2 rounded-lg py-2 ps-3 pe-2 text-sm outline-hidden select-none transition-colors duration-100 ease-[cubic-bezier(0.2,0.8,0.2,1)] data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground data-[state=checked]:font-medium data-[state=checked]:text-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 [&_svg:not([class*='text-'])]:text-muted-foreground",
        className,
      )}
      {...props}
    >
      <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
      <SelectPrimitive.ItemIndicator className="flex shrink-0 items-center text-primary">
        <Check className="size-4 text-primary" />
      </SelectPrimitive.ItemIndicator>
    </SelectPrimitive.Item>
  );
}

function SelectScrollUpButton({
  className,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.ScrollUpButton>) {
  return (
    <SelectPrimitive.ScrollUpButton
      data-slot="select-scroll-up-button"
      className={cn("flex cursor-default items-center justify-center py-1", className)}
      {...props}
    >
      <CaretUp className="size-4" />
    </SelectPrimitive.ScrollUpButton>
  );
}

function SelectScrollDownButton({
  className,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.ScrollDownButton>) {
  return (
    <SelectPrimitive.ScrollDownButton
      data-slot="select-scroll-down-button"
      className={cn("flex cursor-default items-center justify-center py-1", className)}
      {...props}
    >
      <CaretDown className="size-4" />
    </SelectPrimitive.ScrollDownButton>
  );
}

export {
  Select,
  SelectContent,
  SelectItem,
  SelectScrollDownButton,
  SelectScrollUpButton,
  SelectTrigger,
  SelectValue,
};
