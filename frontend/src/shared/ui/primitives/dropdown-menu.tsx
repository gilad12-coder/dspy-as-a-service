"use client";

import * as React from "react";
import { DropdownMenu as DropdownMenuPrimitive } from "radix-ui";

import { cn } from "@/shared/lib/utils";

// Shared panel chrome for the root content and submenus — mirrors
// PopoverContent so menus and popovers read as one family.
const panelClass = cn(
  "origin-(--radix-dropdown-menu-content-transform-origin)",
  "animate-in fade-in-0 zoom-in-95",
  "data-[side=bottom]:slide-in-from-top-2",
  "data-[side=left]:slide-in-from-right-2",
  "data-[side=right]:slide-in-from-left-2",
  "data-[side=top]:slide-in-from-bottom-2",
  "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
  "rounded-xl border border-border/60 bg-background/95 backdrop-blur-xl shadow-lg",
  "outline-none",
);

const rowClass = cn(
  "flex w-full cursor-pointer select-none items-center gap-2 px-3 py-2 text-start text-sm",
  "outline-none data-[highlighted]:bg-accent/60",
  "data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
);

function DropdownMenu({ ...props }: React.ComponentProps<typeof DropdownMenuPrimitive.Root>) {
  return <DropdownMenuPrimitive.Root data-slot="dropdown-menu" {...props} />;
}

function DropdownMenuTrigger({
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Trigger>) {
  return <DropdownMenuPrimitive.Trigger data-slot="dropdown-menu-trigger" {...props} />;
}

function DropdownMenuContent({
  className,
  sideOffset = 6,
  align = "center",
  collisionPadding = 8,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Content>) {
  return (
    <DropdownMenuPrimitive.Portal>
      <DropdownMenuPrimitive.Content
        data-slot="dropdown-menu-content"
        sideOffset={sideOffset}
        align={align}
        collisionPadding={collisionPadding}
        className={cn(panelClass, className)}
        style={{ zIndex: 50 }}
        {...props}
      />
    </DropdownMenuPrimitive.Portal>
  );
}

function DropdownMenuItem({
  className,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Item>) {
  return (
    <DropdownMenuPrimitive.Item
      data-slot="dropdown-menu-item"
      className={cn(rowClass, className)}
      {...props}
    />
  );
}

function DropdownMenuSub({ ...props }: React.ComponentProps<typeof DropdownMenuPrimitive.Sub>) {
  return <DropdownMenuPrimitive.Sub data-slot="dropdown-menu-sub" {...props} />;
}

function DropdownMenuSubTrigger({
  className,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.SubTrigger>) {
  return (
    <DropdownMenuPrimitive.SubTrigger
      data-slot="dropdown-menu-sub-trigger"
      className={cn(rowClass, "data-[state=open]:bg-accent/60", className)}
      {...props}
    />
  );
}

function DropdownMenuSubContent({
  className,
  sideOffset = 6,
  collisionPadding = 8,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.SubContent>) {
  return (
    <DropdownMenuPrimitive.Portal>
      <DropdownMenuPrimitive.SubContent
        data-slot="dropdown-menu-sub-content"
        sideOffset={sideOffset}
        collisionPadding={collisionPadding}
        className={cn(panelClass, className)}
        // Submenus only flip left/right and Radix never clamps the cross
        // axis, so on narrow viewports a fixed-width panel hangs off-screen.
        // Capping to the popper's computed available width lands the panel
        // exactly at the collision padding instead.
        style={{
          zIndex: 50,
          maxWidth: "var(--radix-dropdown-menu-content-available-width)",
        }}
        {...props}
      />
    </DropdownMenuPrimitive.Portal>
  );
}

export {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
};
