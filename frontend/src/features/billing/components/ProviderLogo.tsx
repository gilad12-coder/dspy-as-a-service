"use client";

import * as React from "react";
import { Anthropic, Gemini, Mistral, OpenAI, OpenRouter } from "@lobehub/icons";
import { Plug } from "lucide-react";

/**
 * Full-color brand avatar for a BYOK provider slug.
 *
 * Maps a provider slug onto its @lobehub/icons brand mark (the `.Avatar`
 * variant — a colored, rounded tile). An unknown slug (a custom endpoint, or a
 * provider without a bundled mark) falls back to a neutral plug tile so the row
 * still reads as a connection. `dir="ltr"` keeps the mark upright under RTL.
 */
export function ProviderLogo({ slug, size = 28 }: { slug: string; size?: number }) {
  const logo = renderBrand(slug, size);
  if (logo) {
    return (
      <span dir="ltr" className="inline-flex shrink-0">
        {logo}
      </span>
    );
  }
  return (
    <span
      dir="ltr"
      className="inline-flex shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground"
      style={{ width: size, height: size }}
    >
      <Plug style={{ width: size * 0.5, height: size * 0.5 }} aria-hidden="true" />
    </span>
  );
}

/** Return the brand avatar for a known slug, or null to trigger the fallback tile. */
function renderBrand(slug: string, size: number): React.ReactNode {
  switch (slug) {
    case "openai":
      return <OpenAI.Avatar size={size} />;
    case "anthropic":
      return <Anthropic.Avatar size={size} />;
    case "google":
      return <Gemini.Avatar size={size} />;
    case "mistral":
      return <Mistral.Avatar size={size} />;
    case "openrouter":
      return <OpenRouter.Avatar size={size} />;
    default:
      return null;
  }
}
