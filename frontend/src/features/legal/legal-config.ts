/**
 * Operator-supplied constants for the legal pages (Terms, Privacy).
 *
 * Everything here is a fill-in value that the code cannot know. Replace each
 * TODO with a real, lawyer-reviewed value before relying on these documents.
 * The bracketed defaults render verbatim on the public pages precisely so the
 * gaps are impossible to miss until they are filled.
 */

export const LEGAL_CONFIG = {
  serviceName: "Skynet",
  websiteLabel: "skynetml.com",
  websiteUrl: "https://skynetml.com",

  // Dates shown on both documents. Update lastUpdated whenever the text changes.
  effectiveDate: "August 11, 2026",
  lastUpdated: "August 11, 2026",

  // TODO: replace with the registered legal entity that operates the hosted
  // service. If you trade under "Skynet" with no separate entity, say so here.
  legalEntity: "[Legal entity operating Skynet — replace before launch]",

  // TODO: set up these mailboxes (or replace with real ones). They are
  // referenced from both documents as the contact channels.
  contactEmail: "support@skynetml.com",
  privacyEmail: "privacy@skynetml.com",
  legalEmail: "legal@skynetml.com",

  // TODO: registered business address for legal notices.
  address: "[Registered business address — replace before launch]",

  // TODO: confirm with counsel. Drives the governing-law / venue clause.
  governingLaw: "[Governing jurisdiction — replace before launch]",
  venue: "[Courts/venue — replace before launch]",
} as const;

export const LEGAL_LINKS = {
  terms: "/terms",
  privacy: "/privacy",
} as const;
