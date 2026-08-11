/**
 * Presentational renderer for a legal document (Terms, Privacy).
 *
 * Server component. Renders a bare, self-contained English page (dir="ltr")
 * with its own minimal header and footer, independent of the app chrome. All
 * visible text comes from the passed-in document data or the local English
 * CHROME constants, and every composed string is a single expression (template
 * literal or constant) so no raw JSX text node trips i18next/no-literal-string.
 */

import Link from "next/link";
import { LEGAL_CONFIG } from "./legal-config";
import type { LegalBlock, LegalDocument as LegalDoc } from "./types";

const CHROME = {
  wordmark: "SKYNET",
  homeAria: "Skynet home",
  contents: "Contents",
  lastUpdatedPrefix: "Last updated:",
  backToApp: "Return to Skynet",
} as const;

/** Turn a section heading into a stable URL-fragment id for in-page anchors. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Render one block (paragraph, subheading, or bulleted list) within a section. */
function Block({ block }: { block: LegalBlock }) {
  if (block.kind === "subheading") {
    return <h3 className="mt-6 mb-2 text-base font-semibold text-foreground">{block.text}</h3>;
  }
  if (block.kind === "list") {
    return (
      <ul className="my-3 list-disc space-y-2 ps-6 text-foreground/80">
        {block.items.map((item, i) => (
          <li key={i} className="leading-relaxed">
            {item}
          </li>
        ))}
      </ul>
    );
  }
  return <p className="my-3 leading-relaxed text-foreground/80">{block.text}</p>;
}

/**
 * Render a complete legal document with header, table of contents, numbered
 * sections, and a footer that cross-links the companion document.
 */
export function LegalDocument({
  document,
  related,
}: {
  document: LegalDoc;
  related: { label: string; href: string };
}) {
  const { title, intro, sections } = document;

  return (
    <div dir="ltr" className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/60">
        <div className="mx-auto flex max-w-3xl items-center px-6 py-4">
          <Link
            href="/"
            aria-label={CHROME.homeAria}
            className="text-sm font-bold uppercase tracking-[0.14em] text-foreground"
            style={{ fontFamily: "var(--font-ui)" }}
          >
            {CHROME.wordmark}
          </Link>
        </div>
      </header>

      <article className="mx-auto max-w-3xl px-6 py-10">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">{title}</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          {`${CHROME.lastUpdatedPrefix} ${LEGAL_CONFIG.lastUpdated}`}
        </p>

        <p className="mt-6 leading-relaxed text-foreground/80">{intro}</p>

        <nav
          aria-label={CHROME.contents}
          className="mt-8 rounded-xl border border-border/60 bg-accent/30 p-5"
        >
          <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {CHROME.contents}
          </p>
          <ol className="space-y-1.5 text-sm">
            {sections.map((section, i) => (
              <li key={section.heading}>
                <a
                  href={`#${slugify(section.heading)}`}
                  className="text-foreground/70 hover:text-foreground hover:underline"
                >
                  {`${i + 1}. ${section.heading}`}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <div className="mt-8 space-y-10">
          {sections.map((section, i) => (
            <section key={section.heading} id={slugify(section.heading)} className="scroll-mt-24">
              <h2 className="text-xl font-semibold text-foreground">
                {`${i + 1}. ${section.heading}`}
              </h2>
              <div className="mt-2">
                {section.blocks.map((block, j) => (
                  <Block key={j} block={block} />
                ))}
              </div>
            </section>
          ))}
        </div>

        <footer className="mt-12 border-t border-border/60 pt-6 text-sm text-muted-foreground">
          <p>{`${CHROME.lastUpdatedPrefix} ${LEGAL_CONFIG.lastUpdated}`}</p>
          <p className="mt-3 flex gap-4">
            <Link
              href={related.href}
              className="text-foreground/80 underline underline-offset-2 hover:text-foreground"
            >
              {related.label}
            </Link>
            <Link
              href="/"
              className="text-foreground/80 underline underline-offset-2 hover:text-foreground"
            >
              {CHROME.backToApp}
            </Link>
          </p>
        </footer>
      </article>
    </div>
  );
}
