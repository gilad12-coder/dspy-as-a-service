/**
 * Shared shapes for rendering a legal document (Terms, Privacy) as data.
 *
 * The copy lives as plain string constants (see terms-content.ts /
 * privacy-content.ts) and is rendered via {}-interpolation, which keeps every
 * user-facing string out of raw JSX text nodes and satisfies the repo's
 * i18next/no-literal-string lint rule without routing legal prose through the
 * UI-string i18n catalog.
 */

/** A single rendered element within a section. */
export type LegalBlock =
  | { kind: "paragraph"; text: string }
  | { kind: "subheading"; text: string }
  | { kind: "list"; items: readonly string[] };

/** A numbered top-level section of a legal document. */
export interface LegalSection {
  heading: string;
  blocks: readonly LegalBlock[];
}

/** A complete legal document ready to render. */
export interface LegalDocument {
  title: string;
  intro: string;
  sections: readonly LegalSection[];
}
