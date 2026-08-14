/**
 * Presentational renderer for the public Terms and Privacy documents.
 *
 * Server component. The legal copy stays in the document data files while this
 * renderer provides a standalone, English, print-friendly reading surface.
 */

import Link from "next/link";
import type { Icon } from "@phosphor-icons/react";
import {
  ArrowLeft,
  ArrowRight,
  CalendarBlank,
  CheckCircle,
  Clock,
  Cpu,
  CreditCard,
  Database,
  Envelope,
  FileText,
  Globe,
  Key,
  Lock,
  Scroll,
  ShieldCheck,
  User,
  Users,
} from "@/shared/ui/icons";
import { AnimatedWordmark } from "@/shared/ui/animated-wordmark";
import { Button } from "@/shared/ui/primitives/button";
import { LEGAL_CONFIG } from "./legal-config";
import type { LegalBlock, LegalDocument as LegalDoc } from "./types";
import styles from "./legal-document.module.css";

const CHROME = {
  legal: "Legal",
  homeAria: "Skynet home",
  contents: "On this page",
  contentsAria: "Document sections",
  lastUpdated: "Last updated",
  effective: "Effective",
  readingTime: "Reading time",
  minutes: "min",
  sections: "sections",
  current: "Current document",
  backToApp: "Return to Skynet",
  relatedDocument: "Related document",
  questions: "Questions about this document?",
  contactPrompt: "Contact us and we will help clarify how this document applies to Skynet.",
  termsEyebrow: "Service agreement",
  privacyEyebrow: "Data & privacy",
} as const;

type LegalDocumentKind = "terms" | "privacy";

/** Turn a section heading into a stable URL-fragment id for in-page anchors. */
function slugifyLegalHeading(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Select a semantic icon for a legal section from its heading. */
function iconForSection(heading: string): Icon {
  const value = heading.toLowerCase();

  if (value.includes("account") || value.includes("eligibility") || value.includes("children")) {
    return User;
  }
  if (value.includes("content") || value.includes("information") || value.includes("retention")) {
    return Database;
  }
  if (value.includes("provider") || value.includes("ai output")) return Cpu;
  if (value.includes("key")) return Key;
  if (value.includes("billing") || value.includes("credit") || value.includes("refund")) {
    return CreditCard;
  }
  if (value.includes("security") || value.includes("acceptable")) return ShieldCheck;
  if (value.includes("right") || value.includes("choice")) return Users;
  if (value.includes("cookie")) return Globe;
  if (value.includes("contact")) return Envelope;
  if (value.includes("availability") || value.includes("change")) return Clock;
  if (value.includes("service") || value.includes("transfer")) return Globe;
  if (value.includes("termination") || value.includes("liability")) return Lock;
  if (value.includes("property") || value.includes("governing") || value.includes("dispute")) {
    return Scroll;
  }
  if (value.includes("indemnification") || value.includes("disclaimer")) return ShieldCheck;
  return FileText;
}

/** Estimate the time needed to read the full legal document. */
function readingMinutes(document: LegalDoc): number {
  const blocks = document.sections.flatMap((section) =>
    section.blocks.flatMap((block) => (block.kind === "list" ? block.items : [block.text])),
  );
  const words = [document.intro, ...blocks].join(" ").trim().split(/\s+/).length;
  return Math.max(1, Math.ceil(words / 225));
}

/** Render one paragraph, subheading, or bulleted list within a section. */
function Block({ block }: { block: LegalBlock }) {
  if (block.kind === "subheading") {
    return (
      <h3 className="pt-3 !text-lg font-semibold tracking-tight text-foreground">{block.text}</h3>
    );
  }
  if (block.kind === "list") {
    return (
      <ul className="grid gap-3 text-base leading-7 text-foreground/75">
        {block.items.map((item, index) => (
          <li key={index} className="grid grid-cols-[0.75rem_minmax(0,1fr)] items-start gap-3">
            <span aria-hidden="true" className="mt-[0.68rem] size-1.5 rounded-full bg-[#8C7A6B]" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    );
  }
  return <p className="text-base leading-7 text-foreground/75">{block.text}</p>;
}

/** Render a complete legal document with navigation, metadata, and cross-links. */
export function LegalDocument({
  document,
  kind,
  related,
}: {
  document: LegalDoc;
  kind: LegalDocumentKind;
  related: { label: string; href: string };
}) {
  const { title, intro, sections } = document;
  const isPrivacy = kind === "privacy";
  const DocumentIcon = isPrivacy ? ShieldCheck : Scroll;
  const eyebrow = isPrivacy ? CHROME.privacyEyebrow : CHROME.termsEyebrow;
  const contactEmail = isPrivacy ? LEGAL_CONFIG.privacyEmail : LEGAL_CONFIG.legalEmail;
  const minutes = readingMinutes(document);

  return (
    <div
      dir="ltr"
      className="h-dvh overflow-x-hidden overflow-y-auto bg-[#F5F1EC] text-foreground print:h-auto print:overflow-visible"
    >
      <header className="sticky top-0 z-20 border-b border-border/60 bg-[#FAF8F5]/95 backdrop-blur-md print:static print:bg-white">
        <div className="mx-auto flex min-h-16 w-full max-w-[96rem] items-center justify-between gap-4 px-4 sm:px-8 lg:px-12 xl:px-16">
          <Link
            href="/"
            aria-label={CHROME.homeAria}
            className="flex min-h-11 items-center gap-3 rounded-lg px-1 text-foreground outline-none transition-colors hover:text-[#6F5541] focus-visible:ring-2 focus-visible:ring-[#C8A882]/60"
          >
            <AnimatedWordmark
              size={16}
              className="cursor-pointer"
              autoMorph
              autoMorphDuration={10000}
              morphSpeed={250}
            />
            <span aria-hidden="true" className="h-4 w-px bg-border" />
            <span className="text-sm font-medium text-muted-foreground">{CHROME.legal}</span>
          </Link>

          <Button asChild variant="outline" size="sm">
            <Link href="/" aria-label={CHROME.backToApp}>
              <ArrowLeft className="size-4" aria-hidden="true" />
              <span className="hidden sm:inline">{CHROME.backToApp}</span>
            </Link>
          </Button>
        </div>
      </header>

      <article className="mx-auto w-full max-w-[96rem] px-4 py-6 sm:px-8 sm:py-8 lg:px-12 lg:py-12 xl:px-16">
        <section
          aria-labelledby="legal-document-title"
          className="relative overflow-hidden rounded-[2rem] bg-[#3D2E22] px-6 py-8 text-[#FAF8F5] shadow-[0_24px_70px_-42px_rgba(61,46,34,0.75)] sm:px-10 sm:py-12 lg:px-14 lg:py-16 print:rounded-none print:border-b print:border-black/20 print:bg-white print:text-black print:shadow-none"
        >
          <DocumentIcon
            className="pointer-events-none absolute -bottom-12 right-4 size-56 rotate-[-8deg] text-[#FAF8F5]/[0.055] sm:right-10 sm:size-72 lg:-bottom-16 lg:right-16 lg:size-[22rem] print:hidden"
            weight="thin"
            aria-hidden="true"
          />

          <div className="relative grid gap-10 lg:grid-cols-[minmax(0,1fr)_18rem] lg:items-end">
            <div className="max-w-4xl">
              <div className="flex items-center gap-3 text-[#D9C9B8]">
                <span className="grid size-9 place-items-center rounded-xl border border-[#FAF8F5]/15 bg-[#FAF8F5]/10">
                  <DocumentIcon className="size-4" aria-hidden="true" />
                </span>
                <span className="text-xs font-semibold uppercase tracking-[0.16em]">{eyebrow}</span>
              </div>
              <h1
                id="legal-document-title"
                className={`${styles.title} mt-7 max-w-4xl !text-[clamp(3rem,5vw,4.75rem)] !leading-[0.96] font-bold tracking-[-0.055em]`}
              >
                {title}
              </h1>
              <p className="mt-7 max-w-[72ch] text-base leading-7 text-[#E8DDD1] sm:text-lg sm:leading-8 print:text-black/75">
                {intro}
              </p>
            </div>

            <dl className="grid grid-cols-2 gap-x-5 gap-y-6 border-t border-[#FAF8F5]/15 pt-6 text-sm lg:grid-cols-1 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0 print:border-black/15">
              <div className="flex items-start gap-3">
                <CalendarBlank
                  className="mt-0.5 size-4 shrink-0 text-[#D9C9B8] print:text-black/60"
                  aria-hidden="true"
                />
                <div>
                  <dt className="text-xs uppercase tracking-[0.12em] text-[#BDAA97] print:text-black/55">
                    {CHROME.lastUpdated}
                  </dt>
                  <dd className="mt-1 font-semibold text-[#FAF8F5] print:text-black">
                    {LEGAL_CONFIG.lastUpdated}
                  </dd>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle
                  className="mt-0.5 size-4 shrink-0 text-[#D9C9B8] print:text-black/60"
                  aria-hidden="true"
                />
                <div>
                  <dt className="text-xs uppercase tracking-[0.12em] text-[#BDAA97] print:text-black/55">
                    {CHROME.effective}
                  </dt>
                  <dd className="mt-1 font-semibold text-[#FAF8F5] print:text-black">
                    {LEGAL_CONFIG.effectiveDate}
                  </dd>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Clock
                  className="mt-0.5 size-4 shrink-0 text-[#D9C9B8] print:text-black/60"
                  aria-hidden="true"
                />
                <div>
                  <dt className="text-xs uppercase tracking-[0.12em] text-[#BDAA97] print:text-black/55">
                    {CHROME.readingTime}
                  </dt>
                  <dd className="mt-1 font-semibold text-[#FAF8F5] print:text-black">
                    {`${minutes} ${CHROME.minutes}`}
                  </dd>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <FileText
                  className="mt-0.5 size-4 shrink-0 text-[#D9C9B8] print:text-black/60"
                  aria-hidden="true"
                />
                <div>
                  <dt className="text-xs uppercase tracking-[0.12em] text-[#BDAA97] print:text-black/55">
                    {CHROME.current}
                  </dt>
                  <dd className="mt-1 font-semibold text-[#FAF8F5] print:text-black">
                    {`${sections.length} ${CHROME.sections}`}
                  </dd>
                </div>
              </div>
            </dl>
          </div>
        </section>

        <div className="grid gap-12 py-12 lg:grid-cols-[18rem_minmax(0,1fr)] lg:gap-16 lg:py-16 xl:grid-cols-[20rem_minmax(0,1fr)] xl:gap-24">
          <aside className="print:hidden">
            <nav
              aria-label={CHROME.contentsAria}
              className="border-y border-border/70 py-6 lg:sticky lg:top-24 lg:max-h-[calc(100vh-7rem)] lg:overflow-y-auto lg:overscroll-contain"
            >
              <div className="mb-5 flex items-center justify-between gap-4">
                <p className="text-xs font-bold uppercase tracking-[0.15em] text-foreground">
                  {CHROME.contents}
                </p>
                <span className="rounded-full bg-[#E8DFD4] px-2.5 py-1 font-mono text-[0.6875rem] text-[#6F5541]">
                  {sections.length}
                </span>
              </div>
              <ol className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-1">
                {sections.map((section, index) => (
                  <li key={section.heading}>
                    <a
                      href={`#${slugifyLegalHeading(section.heading)}`}
                      className="group grid min-h-10 grid-cols-[1.75rem_minmax(0,1fr)] items-start gap-2 rounded-lg px-2 py-2 text-sm text-foreground/60 outline-none transition-colors hover:bg-[#EDE7DD] hover:text-foreground focus-visible:ring-2 focus-visible:ring-[#C8A882]/60"
                    >
                      <span className="font-mono text-[0.6875rem] leading-5 text-[#9B8877]">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <span className="leading-5">{section.heading}</span>
                    </a>
                  </li>
                ))}
              </ol>
            </nav>
          </aside>

          <div className="min-w-0 max-w-[78ch]">
            {sections.map((section, index) => {
              const SectionIcon = iconForSection(section.heading);
              return (
                <section
                  key={section.heading}
                  id={slugifyLegalHeading(section.heading)}
                  className="scroll-mt-24 border-b border-border/60 py-10 first:pt-0 last:border-b-0 last:pb-0 print:break-inside-avoid"
                >
                  <div className="grid grid-cols-[2.75rem_minmax(0,1fr)] items-start gap-4 sm:grid-cols-[3rem_minmax(0,1fr)] sm:gap-5">
                    <span className="grid size-11 place-items-center rounded-xl border border-[#D8CCBD] bg-[#EDE7DD] text-[#5C4535] sm:size-12">
                      <SectionIcon className="size-5" aria-hidden="true" />
                    </span>
                    <div className="pt-0.5">
                      <p className="font-mono text-[0.6875rem] font-semibold uppercase tracking-[0.14em] text-[#9B8877]">
                        {String(index + 1).padStart(2, "0")}
                      </p>
                      <h2 className="mt-1 !text-2xl !leading-tight font-bold tracking-[-0.025em] text-foreground sm:!text-[1.75rem]">
                        {section.heading}
                      </h2>
                    </div>
                  </div>
                  <div className="mt-6 grid gap-5 sm:pl-[4.25rem]">
                    {section.blocks.map((block, blockIndex) => (
                      <Block key={blockIndex} block={block} />
                    ))}
                  </div>
                </section>
              );
            })}
          </div>
        </div>

        <footer className="grid overflow-hidden rounded-[2rem] border border-border/70 bg-[#FAF8F5] md:grid-cols-2 print:rounded-none">
          <div className="flex flex-col justify-between gap-8 p-6 sm:p-8 lg:p-10">
            <div>
              <span className="grid size-11 place-items-center rounded-xl bg-[#EDE7DD] text-[#5C4535]">
                <Envelope className="size-5" aria-hidden="true" />
              </span>
              <h2 className="mt-6 text-2xl font-bold tracking-tight text-foreground">
                {CHROME.questions}
              </h2>
              <p className="mt-3 max-w-[52ch] text-base leading-7 text-foreground/65">
                {CHROME.contactPrompt}
              </p>
            </div>
            <Button asChild variant="outline" size="sm" className="w-fit">
              <a href={`mailto:${contactEmail}`}>
                <Envelope className="size-4" aria-hidden="true" />
                {contactEmail}
              </a>
            </Button>
          </div>

          <div className="flex flex-col justify-between gap-8 border-t border-border/70 bg-[#F0EBE4] p-6 sm:p-8 md:border-l md:border-t-0 lg:p-10">
            <div>
              <span className="grid size-11 place-items-center rounded-xl border border-border/70 bg-[#FAF8F5] text-[#5C4535]">
                <FileText className="size-5" aria-hidden="true" />
              </span>
              <p className="mt-6 text-xs font-bold uppercase tracking-[0.15em] text-[#8C7A6B]">
                {CHROME.relatedDocument}
              </p>
              <h2 className="mt-2 text-2xl font-bold tracking-tight text-foreground">
                {related.label}
              </h2>
            </div>
            <Button asChild variant="outline" size="sm" className="group w-fit">
              <Link href={related.href}>
                {related.label}
                <ArrowRight
                  className="size-4 transition-transform duration-200 group-hover:translate-x-0.5 motion-reduce:transition-none"
                  aria-hidden="true"
                />
              </Link>
            </Button>
          </div>
        </footer>
      </article>
    </div>
  );
}
