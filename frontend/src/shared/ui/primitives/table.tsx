"use client";

import * as React from "react";

import { cn } from "@/shared/lib/utils";

function Table({ className, ...props }: React.ComponentProps<"table">) {
  return (
    <div data-slot="table-container" className="relative w-full overflow-x-auto">
      <table
        data-slot="table"
        className={cn("w-full caption-bottom text-sm", className)}
        {...props}
      />
    </div>
  );
}

function TableHeader({ className, ...props }: React.ComponentProps<"thead">) {
  return (
    <thead
      data-slot="table-header"
      className={cn(
        "bg-transparent [&_tr]:border-b [&_tr]:border-border/70 sticky top-0 z-10 bg-background",
        className,
      )}
      {...props}
    />
  );
}

function TableBody({ className, ...props }: React.ComponentProps<"tbody">) {
  return (
    <tbody
      data-slot="table-body"
      className={cn("[&_tr:last-child]:border-0", className)}
      {...props}
    />
  );
}

function TableRow({ className, ...props }: React.ComponentProps<"tr">) {
  return (
    <tr
      data-slot="table-row"
      className={cn("border-b border-border/60 data-[state=selected]:bg-muted/75", className)}
      {...props}
    />
  );
}

function TableHead({ className, scope = "col", ...props }: React.ComponentProps<"th">) {
  return (
    <th
      data-slot="table-head"
      scope={scope}
      className={cn(
        "h-12 px-2 text-start align-middle text-[0.75rem] font-semibold whitespace-nowrap text-muted-foreground [&:has([role=checkbox])]:pe-0 [&>[role=checkbox]]:translate-y-[2px]",
        className,
      )}
      {...props}
    />
  );
}

function TableCell({ className, ...props }: React.ComponentProps<"td">) {
  // ``overflow-hidden text-ellipsis`` (+ the existing ``whitespace-nowrap``)
  // makes every cell width-truncate dynamically — when the caller sets a
  // ``max-w-*`` on the cell, overflowing content gets the ``…`` rendered
  // by the browser at the exact column boundary instead of by a manual
  // ``slice(0, N)`` upstream. Cells that need to wrap (long copy, code
  // blocks) opt in with ``whitespace-normal break-words``; cells with no
  // ``max-w`` simply expand to content and never overflow.
  return (
    <td
      data-slot="table-cell"
      className={cn(
        "p-2.5 align-middle whitespace-nowrap overflow-hidden text-ellipsis [&:has([role=checkbox])]:pe-0 [&>[role=checkbox]]:translate-y-[2px]",
        className,
      )}
      {...props}
    />
  );
}

export { Table, TableHeader, TableBody, TableHead, TableRow, TableCell };
