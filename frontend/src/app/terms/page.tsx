/** Public Terms of Service page at /terms. */

import type { Metadata } from "next";
import { LegalDocument, LEGAL_LINKS, TERMS_OF_SERVICE } from "@/features/legal";

export const metadata: Metadata = {
  title: "Terms of Service",
  description: "The terms that govern your use of the hosted Skynet service.",
  alternates: { canonical: LEGAL_LINKS.terms },
  robots: { index: true, follow: true },
};

export default function TermsPage() {
  return (
    <LegalDocument
      document={TERMS_OF_SERVICE}
      kind="terms"
      related={{ label: "Privacy Policy", href: LEGAL_LINKS.privacy }}
    />
  );
}
