"use client";

import * as React from "react";
import { useRef, useCallback } from "react";
import { cn } from "@/shared/lib/utils";

function Card({ className, ...props }: React.ComponentProps<"div">) {
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const el = cardRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    el.style.setProperty("--spotlight-x", `${x}px`);
    el.style.setProperty("--spotlight-y", `${y}px`);
  }, []);

  const handleMouseLeave = useCallback(() => {
    const el = cardRef.current;
    if (!el) return;
    el.style.removeProperty("--spotlight-x");
    el.style.removeProperty("--spotlight-y");
  }, []);

  return (
    <div
      ref={cardRef}
      data-slot="card"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className={cn(
        "group/card relative flex flex-col gap-4 rounded-2xl",
        "py-5 text-card-foreground",
        "bg-gradient-to-b from-white/95 to-[#F8F4EF]",
        "border border-[#DDD4C8]/50",
        "shadow-[0_1px_3px_rgba(28,22,18,0.03),0_4px_16px_rgba(28,22,18,0.025)]",
        "backdrop-blur-sm",
        "transition-[border-color,box-shadow] duration-300 ease-[cubic-bezier(0.2,0.8,0.2,1)]",
        className,
      )}
      {...props}
    />
  );
}

function CardHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-header"
      className={cn(
        "@container/card-header grid auto-rows-min grid-rows-[auto_auto] items-start gap-2.5 px-6 [.border-b]:pb-6",
        className,
      )}
      {...props}
    />
  );
}

function CardTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-title"
      className={cn("leading-none font-semibold tracking-tight text-foreground", className)}
      {...props}
    />
  );
}

function CardDescription({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-description"
      className={cn("text-sm text-muted-foreground", className)}
      {...props}
    />
  );
}

function CardContent({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="card-content" className={cn("px-6 text-sm", className)} {...props} />;
}

export { Card, CardHeader, CardTitle, CardDescription, CardContent };
