/** Public Privacy Policy page at /privacy. */

import type { Metadata } from "next";
import { LegalDocument, LEGAL_LINKS, PRIVACY_POLICY } from "@/features/legal";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "How Skynet collects, uses, and shares your personal information.",
  alternates: { canonical: LEGAL_LINKS.privacy },
  robots: { index: true, follow: true },
};

export default function PrivacyPage() {
  return (
    <LegalDocument
      document={PRIVACY_POLICY}
      related={{ label: "Terms of Service", href: LEGAL_LINKS.terms }}
    />
  );
}
