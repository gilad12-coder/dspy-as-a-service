import { msg } from "@/shared/lib/messages";
import type { TaggerConfig } from "./types";
import { isBinaryNo, isBinaryYes } from "./types";

function resolveCategoryLabel(categories: TaggerConfig["categories"], value: unknown): string {
  const raw = String(value ?? "").trim();
  if (!raw) return "";

  const categoryList = categories ?? [];
  const exactId = categoryList.find((category) => String(category.id) === raw);
  if (exactId) return exactId.label;

  const exactLabel = categoryList.find(
    (category) => category.label.trim().toLocaleLowerCase() === raw.toLocaleLowerCase(),
  );
  if (exactLabel) return exactLabel.label;

  const numericIndex = Number(raw);
  if (Number.isInteger(numericIndex) && numericIndex >= 1) {
    const oneBasedCategory = categoryList[numericIndex - 1];
    if (oneBasedCategory) return oneBasedCategory.label;
  }

  return raw;
}

/** Format a stored annotation exactly as it should appear in the tagger UI. */
export function formatTaggerLabel(config: TaggerConfig, value: unknown): string {
  if (value === undefined || value === null) return "";

  if (config.mode === "multiclass") {
    const values = Array.isArray(value) ? value : [value];
    return values
      .map((item) => resolveCategoryLabel(config.categories, item))
      .filter(Boolean)
      .join(", ");
  }

  if (config.mode === "binary") {
    const normalized = String(value).trim();
    if (isBinaryYes(normalized)) return msg("tagger.assist.label.yes");
    if (isBinaryNo(normalized)) return msg("tagger.assist.label.no");
  }

  return String(value);
}
