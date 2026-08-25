/**
 * Operator-supplied constants for the legal pages (Terms, Privacy).
 *
 * These are the real, operator-specific values the code cannot derive; they
 * render verbatim on the public /terms and /privacy pages. Keep them accurate
 * and bump lastUpdated whenever the document text changes. The documents
 * themselves are launch-ready drafts, not legal advice — have counsel review
 * them before you rely on them.
 */

export const LEGAL_CONFIG = {
  serviceName: "Skynet",
  websiteLabel: "skynetml.com",
  websiteUrl: "https://skynetml.com",

  // Dates shown on both documents. Update lastUpdated whenever the text changes.
  effectiveDate: "August 11, 2026",
  lastUpdated: "August 14, 2026",

  // Skynet is operated by an individual (sole proprietor); this legal name is
  // the operating party and, for the Privacy Policy, the data controller.
  legalEntity: "Gilad Morad",

  // Contact channels referenced from both documents. These mailboxes must be
  // live and monitored before launch — privacy@ in particular, since it is the
  // address for GDPR/CCPA requests — or messages to them bounce.
  contactEmail: "support@skynetml.com",
  privacyEmail: "privacy@skynetml.com",
  legalEmail: "legal@skynetml.com",

  // Governing-law / venue clause.
  governingLaw: "the State of New York, United States",
  venue: "New York County, New York",
} as const;

export const LEGAL_LINKS = {
  terms: "/terms",
  privacy: "/privacy",
} as const;
