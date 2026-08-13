"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "radix-ui";

import { cn } from "@/shared/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
  useTooltipTriggerContext,
} from "@/shared/ui/primitives/tooltip";

const buttonVariants = cva(
  "inline-flex shrink-0 select-none items-center justify-center gap-2 rounded-md border border-transparent text-sm font-medium whitespace-nowrap outline-none cursor-pointer transform-gpu transition-[transform,background-color,color,border-color,box-shadow,opacity] duration-75 ease-out active:scale-[0.97] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 motion-reduce:transform-none motion-reduce:transition-none aria-invalid:border-destructive aria-invalid:ring-destructive/20 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive:
          "bg-destructive text-white shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/20",
        outline:
          "border-input bg-background shadow-xs hover:bg-accent hover:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground shadow-xs hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "h-auto border-none px-0 py-0 text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        xs: "h-7 gap-1 rounded-md px-2.5 text-xs has-[>svg]:px-2 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-8 gap-1.5 rounded-md px-3 has-[>svg]:px-2.5",
        lg: "h-10 rounded-md px-5 has-[>svg]:px-4",
        pill: "h-10 rounded-full px-7 text-sm font-semibold has-[>svg]:px-5 hover:shadow-[0_0_16px_rgba(61,46,34,0.2)]",
        "pill-sm":
          "h-8 rounded-full px-5 text-xs font-semibold has-[>svg]:px-3.5 hover:shadow-[0_0_12px_rgba(61,46,34,0.15)]",
        "pill-lg":
          "h-12 rounded-full px-9 text-base font-semibold has-[>svg]:px-6 hover:shadow-[0_0_20px_rgba(61,46,34,0.25)]",
        icon: "size-9 rounded-md",
        "icon-xs": "size-7 rounded-md [&_svg:not([class*='size-'])]:size-3.5",
        "icon-sm": "size-8 rounded-md",
        "icon-lg": "size-10 rounded-md",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

type ButtonVariantProps = VariantProps<typeof buttonVariants>;
type ButtonSize = NonNullable<ButtonVariantProps["size"]>;
type IconButtonSize = Extract<ButtonSize, `icon${string}`>;
type ButtonProps = Omit<React.ComponentProps<"button">, "aria-label"> &
  Omit<ButtonVariantProps, "size"> & {
    asChild?: boolean;
  } & (
    | {
        size: IconButtonSize;
        "aria-label": string;
      }
    | {
        size?: Exclude<ButtonSize, IconButtonSize> | null;
        "aria-label"?: string;
      }
  );

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  "aria-label": ariaLabel,
  ...props
}: ButtonProps) {
  const hasExplicitTooltip = useTooltipTriggerContext();
  const Comp = asChild ? Slot.Root : "button";

  const button = (
    <Comp
      {...(!asChild ? { type: "button" as const } : {})}
      data-slot="button"
      data-variant={variant}
      data-size={size}
      aria-label={ariaLabel}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );

  if (typeof size !== "string" || !size.startsWith("icon") || hasExplicitTooltip) {
    return button;
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>{button}</TooltipTrigger>
        <TooltipContent>{ariaLabel}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export { Button, buttonVariants };
