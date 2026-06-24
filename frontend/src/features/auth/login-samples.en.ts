/**
 * English titles for the decorative login halo, mirroring `login-samples.ts`.
 *
 * Order-coupled: `HALO_CARDS_EN[i]` is the English label for `HALO_CARDS[i]`.
 * Keep this array the same length and order as `HALO_CARDS` (LoginHalo zips the
 * two by index). Positions/rotation live only on the Hebrew source; this file
 * carries just the localized strings, the same way `messages.en.ts` overlays
 * `messages.ts`.
 */

export const HALO_CARDS_EN: readonly string[] = [
  // Top band — front row
  "Intent classification",
  "Meeting summary",
  "Document tagging",
  "Language detection",
  "Content moderation",
  "Query rewriting",
  "Entity linking",

  // Top band — middle row
  "Entity extraction",
  "Sentiment analysis",
  "Keyword extraction",
  "Spam detection",
  "Grammar correction",

  // Top band — back row
  "Quote extraction",
  "Part-of-speech tagging",

  // Left wing — outer column
  "Relevance ranking",
  "Question answering",
  "Product matching",
  "Contradiction detection",

  // Left wing — inner column
  "Quality scoring",
  "Topic extraction",

  // Right wing — outer column
  "Tool use",
  "Answer ranking",
  "Fraud detection",
  "Mathematical reasoning",

  // Right wing — inner column
  "Grounded answering",
  "Fact checking",

  // Bottom band — back row
  "Query expansion",
  "JSON extraction",

  // Bottom band — middle row
  "Product attribute extraction",
  "Relation extraction",
  "Contact extraction",
  "Text-to-SQL",
  "Toxicity detection",
  "Document summarization",

  // Bottom band — front row
  "Date extraction",
  "Category classification",
  "Purchase-intent detection",
  "Multi-step retrieval",
  "Ticket classification",
  "Ticket routing",
];
