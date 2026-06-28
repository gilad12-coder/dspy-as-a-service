import type { Metadata } from "next";
import { msg } from "@/shared/lib/messages";

export const metadata: Metadata = {
  title: "Credits",
  description: msg("billing.upgrade.meta_description"),
};

export default function UpgradeLayout({ children }: { children: React.ReactNode }) {
  return children;
}
