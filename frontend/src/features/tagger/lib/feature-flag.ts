/**
 * Build-time kill switch for the tagger's AI co-tagging assist, following the
 * generalist-agent flag pattern. Defaults ON; set
 * ``NEXT_PUBLIC_FEATURE_TAGGER_ASSIST=0`` to ship the plain manual tagger.
 */
export function isTaggerAssistEnabled(): boolean {
  const raw = process.env.NEXT_PUBLIC_FEATURE_TAGGER_ASSIST;
  if (raw === undefined || raw === "") return true;
  const v = raw.toLowerCase().trim();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}
