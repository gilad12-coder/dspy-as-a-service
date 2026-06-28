"use client";

import * as React from "react";
import Link from "next/link";
import { ChevronRight, KeyRound, LogOut, Settings, Sparkles } from "lucide-react";
import { signOut, useSession } from "next-auth/react";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { dirForLocale } from "@/shared/lib/locale";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/ui/primitives/popover";
import { useSettingsModal } from "@/features/settings";
import { useCredits } from "@/features/billing";

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
 * Account menu — the avatar-triggered dropdown in the header.
 *
 * Structure mirrors a familiar account menu: a profile header row (avatar, name,
 * plan badge) that opens the account settings tab, a separator, the primary
 * destinations (Upgrade, Settings, API keys), then a separator and Log out. All
 * targets are real Skynet surfaces — `/upgrade`, the settings modal, and its API
 * (BYOK) tab. Logical alignment and a flipped chevron keep it correct in RTL.
 */
export function AccountMenu() {
  const { data: session } = useSession();
  const { locale } = useLocale();
  const dir = dirForLocale(locale);
  const { setOpen: setSettingsOpen, openTo } = useSettingsModal();
  const { wallet } = useCredits();
  const [open, setOpen] = React.useState(false);

  if (!session?.user) return null;

  const name = session.user.name ?? session.user.email ?? "";
  const planLabel = wallet.premiumActive
    ? msg("app.shell.account.plan_premium")
    : msg("app.shell.account.plan_free");

  const close = () => setOpen(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={msg("app.shell.account.aria")}
          className="rounded-full transition-transform duration-150 active:scale-95 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
        >
          <Avatar image={session.user.image} label={name} size={28} />
        </button>
      </PopoverTrigger>

      <PopoverContent align="end" dir={dir} className="w-64 p-1.5">
        <button
          type="button"
          onClick={() => {
            close();
            openTo("account");
          }}
          className={cn(MENU_ITEM, "gap-2.5 py-2")}
        >
          <Avatar image={session.user.image} label={name} size={32} />
          <span className="flex min-w-0 flex-1 flex-col text-start">
            <span dir="auto" className="truncate font-medium leading-tight text-foreground">
              {name}
            </span>
            <span className="text-xs text-muted-foreground">{planLabel}</span>
          </span>
          <ChevronRight className="size-4 shrink-0 text-muted-foreground rtl:-scale-x-100" aria-hidden="true" />
        </button>

        <div role="separator" className="my-1 h-px bg-border/60" />

        <Link href="/upgrade" onClick={close} className={MENU_ITEM}>
          <Sparkles className="size-4 text-[#C8A882]" aria-hidden="true" />
          {msg("app.shell.account.upgrade")}
        </Link>
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
        <button
          type="button"
          onClick={() => {
            close();
            openTo("api");
          }}
          className={MENU_ITEM}
        >
          <KeyRound className="size-4 text-muted-foreground" aria-hidden="true" />
          {msg("app.shell.account.keys")}
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
          <LogOut className="size-4 text-muted-foreground rtl:-scale-x-100" aria-hidden="true" />
          {msg("app.shell.logout")}
        </button>
      </PopoverContent>
    </Popover>
  );
}
