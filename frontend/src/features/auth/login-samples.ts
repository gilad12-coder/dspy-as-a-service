/**
 * Decorative sample artifacts for the login backdrop.
 *
 * These are FAKE, illustrative task labels — never real user data — used purely
 * as the scattered "product halo" behind the sign-in panel. Each card is a
 * finished task shown by its name, so the halo reads like a wall of Skynet's own
 * completed runs without exposing any numbers. The label is a UI catalog key
 * (``auth.halo.*``), so the backdrop translates with the rest of the UI through
 * the active locale's catalog and fallback chain — not a hardcoded string.
 *
 * Positions are hand-placed around the edges to keep a clean centre; the wing
 * columns hug the form's left/right so the halo frames it rather than drifting
 * out to the viewport edges. The full halo shows from the `md` breakpoint up;
 * below it there isn't room to scatter cards around a near-full-width panel, so a
 * curated few carry `mobilePos` to frame the login on small portrait screens,
 * and the rest drop away (see LoginHalo).
 */

import type { MessageKey } from "@/shared/lib/generated/ui-catalog";

export interface HaloCard {
  /** Catalog key (``auth.halo.*``) resolved to the active locale at render time. */
  key: MessageKey;
  pos: { top?: string; bottom?: string; left?: string; right?: string };
  rot: number;
  /** When set, the card also frames the login on small portrait screens, placed here. */
  mobilePos?: { top?: string; bottom?: string; left?: string; right?: string };
}

export const HALO_CARDS: HaloCard[] = [
  // Top band — front row
  {
    key: "auth.halo.intent_classification",
    pos: { top: "-3%", left: "1%" },
    rot: -5,
    mobilePos: { top: "6%", left: "4%" },
  },
  { key: "auth.halo.meeting_summary", pos: { top: "-4%", left: "15%" }, rot: 3 },
  { key: "auth.halo.document_tagging", pos: { top: "-2%", left: "29%" }, rot: -4 },
  {
    key: "auth.halo.language_detection",
    pos: { top: "-3%", left: "43%" },
    rot: 4,
    mobilePos: { top: "13%", right: "5%" },
  },
  { key: "auth.halo.content_moderation", pos: { top: "-2%", left: "57%" }, rot: -3 },
  { key: "auth.halo.query_rewriting", pos: { top: "-4%", left: "71%" }, rot: 5 },
  { key: "auth.halo.entity_linking", pos: { top: "-3%", right: "2%" }, rot: -5 },

  // Top band — middle row
  { key: "auth.halo.entity_extraction", pos: { top: "9%", left: "7%" }, rot: 4 },
  {
    key: "auth.halo.sentiment_analysis",
    pos: { top: "11%", left: "26%" },
    rot: -3,
    mobilePos: { top: "20%", left: "22%" },
  },
  { key: "auth.halo.keyword_extraction", pos: { top: "9%", left: "45%" }, rot: 5 },
  { key: "auth.halo.spam_detection", pos: { top: "11%", left: "64%" }, rot: -4 },
  { key: "auth.halo.grammar_correction", pos: { top: "9%", left: "81%" }, rot: 3 },

  // Top band — back row
  { key: "auth.halo.quote_extraction", pos: { top: "19%", left: "6%" }, rot: -4 },
  { key: "auth.halo.difficulty_ranking", pos: { top: "20%", left: "47%" }, rot: 5 },
  { key: "auth.halo.pos_tagging", pos: { top: "19%", left: "80%" }, rot: -3 },

  // Left wing — outer column
  { key: "auth.halo.relevance_ranking", pos: { top: "30%", left: "2%" }, rot: 5 },
  { key: "auth.halo.question_answering", pos: { top: "44%", left: "2%" }, rot: 6 },
  { key: "auth.halo.product_matching", pos: { top: "58%", left: "3%" }, rot: 4 },
  { key: "auth.halo.contradiction_detection", pos: { top: "70%", left: "2%" }, rot: -3 },

  // Left wing — inner column
  { key: "auth.halo.quality_scoring", pos: { top: "37%", left: "17%" }, rot: -4 },
  { key: "auth.halo.topic_extraction", pos: { top: "64%", left: "17%" }, rot: -3 },

  // Right wing — outer column
  { key: "auth.halo.tool_use", pos: { top: "30%", right: "4%" }, rot: -5 },
  { key: "auth.halo.answer_ranking", pos: { top: "44%", right: "5%" }, rot: -6 },
  { key: "auth.halo.fraud_detection", pos: { top: "58%", right: "6%" }, rot: -4 },
  { key: "auth.halo.mathematical_reasoning", pos: { top: "70%", right: "5%" }, rot: 5 },

  // Right wing — inner column
  { key: "auth.halo.grounded_answering", pos: { top: "37%", right: "17%" }, rot: 4 },
  {
    key: "auth.halo.fact_checking",
    pos: { top: "64%", right: "17%" },
    rot: 5,
    mobilePos: { bottom: "20%", right: "20%" },
  },

  // Bottom band — back row
  { key: "auth.halo.query_expansion", pos: { bottom: "19%", left: "6%" }, rot: 4 },
  { key: "auth.halo.confidence_estimation", pos: { bottom: "20%", left: "47%" }, rot: -3 },
  {
    key: "auth.halo.json_extraction",
    pos: { bottom: "19%", left: "80%" },
    rot: 4,
    mobilePos: { bottom: "6%", right: "4%" },
  },

  // Bottom band — middle row
  { key: "auth.halo.product_attribute_extraction", pos: { bottom: "10%", left: "9%" }, rot: -4 },
  { key: "auth.halo.relation_extraction", pos: { bottom: "11%", left: "25%" }, rot: 4 },
  { key: "auth.halo.contact_extraction", pos: { bottom: "9%", left: "41%" }, rot: -5 },
  { key: "auth.halo.text_to_sql", pos: { bottom: "11%", left: "57%" }, rot: 3 },
  { key: "auth.halo.toxicity_detection", pos: { bottom: "9%", left: "72%" }, rot: -4 },
  {
    key: "auth.halo.document_summarization",
    pos: { bottom: "11%", left: "85%" },
    rot: 4,
    mobilePos: { bottom: "13%", left: "5%" },
  },

  // Bottom band — front row
  { key: "auth.halo.date_extraction", pos: { bottom: "-3%", left: "7%" }, rot: 4 },
  { key: "auth.halo.category_classification", pos: { bottom: "2%", left: "23%" }, rot: 5 },
  { key: "auth.halo.purchase_intent_detection", pos: { bottom: "-2%", left: "40%" }, rot: -3 },
  { key: "auth.halo.multi_step_retrieval", pos: { bottom: "3%", left: "57%" }, rot: 4 },
  { key: "auth.halo.ticket_classification", pos: { bottom: "-3%", left: "73%" }, rot: 5 },
  { key: "auth.halo.ticket_routing", pos: { bottom: "-2%", right: "4%" }, rot: 5 },
];
