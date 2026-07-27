"use client";

import * as React from "react";
import { LogOut, MoreHorizontal, Settings, Sparkles } from "lucide-react";
import { signOut, useSession } from "next-auth/react";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { dirForLocale } from "@/shared/lib/locale";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/ui/primitives/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { useSettingsModal } from "@/features/settings";

/** Two-letter monogram from a display name or email — the avatar fallback when there's no image. */
function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  const first = parts[0] ?? "";
  if (parts.length >= 2) {
    const second = parts[1] ?? "";
    return ((first[0] ?? "") + (second[0] ?? "")).toUpperCase();
  }
  return name.slice(0, 2).toUpperCase();
}

function Avatar({ image, label, size }: { image?: string | null; label: string; size: number }) {
  if (image) {
    return (
      // Avatars come from arbitrary OAuth providers (Google, GitHub, Auth0); a
      // plain <img> avoids per-host next/image remotePatterns config.
      <img
        src={image}
        alt=""
        width={size}
        height={size}
        referrerPolicy="no-referrer"
        className="shrink-0 rounded-full object-cover"
        style={{ width: size, height: size }}
      />
    );
  }
  return (
    <span
      aria-hidden="true"
      className="grid shrink-0 place-items-center rounded-full bg-[#3D2E22] font-semibold text-[#FAF8F5]"
      style={{ width: size, height: size, fontSize: size * 0.4 }}
    >
      {initials(label)}
    </span>
  );
}

const MENU_ITEM =
  "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-foreground transition-colors duration-150 cursor-pointer hover:bg-accent focus-visible:outline-none focus-visible:bg-accent";

/**
 * Account menu — the profile button anchored at the foot of the sidebar.
 *
 * Mirrors a familiar account button (avatar, name, plan badge) the way ChatGPT
 * sits one at the bottom of its rail. Clicking it opens a menu *upward* with a
 * profile header row (→ account settings tab), a separator, the primary
 * destinations (Upgrade, Settings, API keys), then a separator and Log out — so
 * Settings lives inside this menu rather than as its own row. When the rail is
 * collapsed the trigger shrinks to the avatar alone, with a tooltip naming it.
 * All targets are real Skynet surfaces; logical alignment and a flipped chevron
 * keep it correct in RTL.
 */
export function AccountMenu({ collapsed = false }: { collapsed?: boolean }) {
  const { data: session } = useSession();
  const { locale } = useLocale();
  const dir = dirForLocale(locale);
  const isRtl = dir === "rtl";
  // The collapsed-rail tooltip reads toward the content area: right in LTR, left in RTL.
  const tooltipSide = isRtl ? "left" : "right";
  const { setOpen: setSettingsOpen, openTo: openSettingsTo } = useSettingsModal();
  const [open, setOpen] = React.useState(false);

  if (!session?.user) return null;

  const name = session.user.name ?? session.user.email ?? "";
  const email = session.user.email ?? "";
  // Skip the email line when there's no display name — `name` already falls back to it.
  const showEmail = Boolean(email) && email !== name;

  const close = () => setOpen(false);

  const trigger = collapsed ? (
    <button
      type="button"
      aria-label={msg("app.shell.account.aria")}
      className="rounded-full transition-transform duration-150 active:scale-95 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
    >
      <Avatar image={session.user.image} label={name} size={28} />
    </button>
  ) : (
    <button
      type="button"
      aria-label={msg("app.shell.account.aria")}
      className="group flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-start transition-colors duration-200 hover:bg-sidebar-accent/40 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
    >
      <Avatar image={session.user.image} label={name} size={28} />
      {/* Physical edge from the UI dir, not logical `text-start`: the name keeps
          dir="auto" so it shapes and truncates by its own script, which would make
          `start` resolve to the name's direction and split it from the email line
          onto the opposite edge in a mixed-script UI (e.g. a Latin name, Hebrew UI). */}
      <span className={cn("flex min-w-0 flex-1 flex-col", isRtl ? "text-right" : "text-left")}>
        <span
          dir="auto"
          className="truncate text-sm font-medium leading-tight text-sidebar-foreground"
        >
          {name}
        </span>
        {showEmail && (
          <span dir="ltr" className="truncate text-xs leading-tight text-muted-foreground">
            {email}
          </span>
        )}
      </span>
      <MoreHorizontal
        className="size-4 shrink-0 text-muted-foreground/50 transition-colors group-hover:text-muted-foreground"
        aria-hidden="true"
      />
    </button>
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      {collapsed ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <PopoverTrigger asChild>{trigger}</PopoverTrigger>
          </TooltipTrigger>
          <TooltipContent side={tooltipSide}>{msg("app.shell.account.aria")}</TooltipContent>
        </Tooltip>
      ) : (
        <PopoverTrigger asChild>{trigger}</PopoverTrigger>
      )}

      <PopoverContent
        side="top"
        align="start"
        dir={dir}
        className="w-64 p-1.5"
        collisionPadding={12}
      >
        {/* Identity header — who you're signed in as. Deliberately non-interactive:
            account and general settings both live in the items below, so a click
            target here would just duplicate the Settings row. Surfaces the email,
            the way ChatGPT's account menu does. */}
        <div className="flex items-center gap-2.5 px-2.5 py-2">
          <Avatar image={session.user.image} label={name} size={32} />
          {/* Physical edge from the UI dir — see the trigger above for why `text-start`
              can't be used while the name carries dir="auto". */}
          <span className={cn("flex min-w-0 flex-1 flex-col", isRtl ? "text-right" : "text-left")}>
            <span dir="auto" className="truncate font-medium leading-tight text-foreground">
              {name}
            </span>
            {showEmail && (
              <span dir="ltr" className="truncate text-xs text-muted-foreground">
                {email}
              </span>
            )}
          </span>
        </div>

        <div role="separator" className="my-1 h-px bg-border/60" />

        <button
          type="button"
          onClick={() => {
            close();
            openSettingsTo("billing");
          }}
          className={MENU_ITEM}
        >
          <Sparkles className="size-4 text-[#C8A882]" aria-hidden="true" />
          {msg("app.shell.account.upgrade")}
        </button>
        <button
          type="button"
          onClick={() => {
            close();
            setSettingsOpen(true);
          }}
          className={MENU_ITEM}
        >
          <Settings className="size-4 text-muted-foreground" aria-hidden="true" />
          {msg("app.shell.account.settings")}
        </button>
        <div role="separator" className="my-1 h-px bg-border/60" />

        <button
          type="button"
          onClick={() => {
            close();
            void signOut({ callbackUrl: "/login" });
          }}
          className={MENU_ITEM}
        >
          {/* Point the arrow "outward" toward the sidebar's edge so it reads as
              leaving: the rail sits on the left in LTR (arrow ←, flipped from the
              lucide default) and on the right in RTL (arrow →, the default). Derive
              the flip from `isRtl` rather than the `rtl:` Tailwind variant — this
              menu is a portaled popover where the variant doesn't fire reliably, and
              the rest of the file drives direction the same way. */}
          <LogOut
            className={cn("size-4 text-muted-foreground", !isRtl && "-scale-x-100")}
            aria-hidden="true"
          />
          {msg("app.shell.logout")}
        </button>
      </PopoverContent>
    </Popover>
  );
}
