/**
 * Centralized tooltip copy.
 *
 * Tooltips that describe the same concept should share a single string.
 * Call sites import `tip(key)` and pass the result to <HelpTip text={...}>.
 *
 *     import { tip } from "@/shared/lib/tooltips";
 *     <HelpTip text={tip("score.baseline")}>{TERMS.baselineScore}</HelpTip>
 *
 * Keys are grouped by domain concept, not by feature slice — the same
 * definition of "baseline score" should read identically on the overview
 * tab, the pair detail view, and the compare page.
 *
 * `TOOLTIPS` below is the Hebrew base; the English copy is the `TOOLTIPS_EN`
 * overlay in `tooltips.en.ts`. `tip()` resolves against the active locale, so
 * any new key MUST be added to BOTH files — an English-missing key falls back
 * to Hebrew and leaks into the English UI.
 */

import { fallbackChain, type Locale } from "@/shared/lib/locale";
import { getActiveLocale } from "@/shared/lib/runtime-locale";
// The static generated Hebrew glossary, NOT the locale-aware `TERMS` proxy:
// this base catalog is Hebrew by contract (tip() falls back to it from every
// locale), and the proxy resolved at module scope would freeze whatever locale
// happened to be active at bundle evaluation into these sentences.
import { TERMS as TERMS_HE } from "@/shared/lib/generated/i18n-catalog";
import { TOOLTIPS_EN } from "@/shared/lib/tooltips.en";

export const TOOLTIPS = {
  "score.baseline": `${TERMS_HE.baselineScore} לפני ${TERMS_HE.optimization}: איך ה${TERMS_HE.program} הצליחה בלי פרומפט משופר או דוגמאות נבחרות`,
  "score.optimized": `${TERMS_HE.optimizedScore} אחרי ${TERMS_HE.optimization}: איך ה${TERMS_HE.program} הצליחה עם הפרומפט והדוגמאות שנבחרו`,
  "score.improvement": `הפער בין ה${TERMS_HE.optimizedScore} ל${TERMS_HE.baselineScore}. ככל שהוא גדול יותר, ה${TERMS_HE.optimization} שיפרה יותר את התוצאה`,
  "score.progression": `איך ה${TERMS_HE.score} השתנה מניסיון לניסיון בזמן שה${TERMS_HE.optimizer} חיפש פרומפט טוב יותר`,

  "lm.calls_count": `מספר הקריאות ל${TERMS_HE.model} השפה במהלך ה${TERMS_HE.optimization}`,
  "lm.avg_response_time": `הזמן הממוצע שלקח ל${TERMS_HE.model} לענות לכל קריאה`,

  "lm_activity.section": `פעילות מודלי השפה לפי שלב — כמה קריאות בוצעו וכמה זמן לקחו, מה${TERMS_HE.generationModelShort} וממודל הרפלקציה בנפרד`,
  "lm_activity.stage.baseline": `קריאות שבוצעו בעת מדידת ה${TERMS_HE.baselineScore} — לפני שה${TERMS_HE.optimizer} התחיל לפעול`,
  "lm_activity.stage.training": `קריאות שבוצעו במהלך ה${TERMS_HE.optimization} עצמה — כשהאופטימייזר בנה מועמדים לפרומפט`,
  "lm_activity.stage.evaluation": `קריאות שבוצעו בעת מדידת ה${TERMS_HE.optimizedScore} — אחרי שה${TERMS_HE.optimization} הסתיימה`,
  "lm_activity.column.generation": `קריאות שבוצעו ל${TERMS_HE.generationModel} — המודל שמייצר תשובות`,
  "lm_activity.column.reflection": `קריאות שבוצעו ל${TERMS_HE.reflectionModel} — המודל שמנתח שגיאות ומציע שיפורים`,
  "lm_activity.cell.calls": "מספר הקריאות שבוצעו בשלב הזה",
  "lm_activity.cell.avg_ms": "הזמן הממוצע לקריאה בשלב הזה",
  "lm_activity.total_row": "סך הכול הקריאות והזמן הממוצע על פני כל השלבים",

  "model.generation": `ה${TERMS_HE.model} שמייצר את התשובה בפועל בזמן ה${TERMS_HE.optimizationTypeRun}`,
  "model.reflection": `ה${TERMS_HE.model} שבודק טעויות ומציע איך לשפר את ההנחיות במהלך ה${TERMS_HE.optimization}`,

  "data.split_explanation": `ה${TERMS_HE.dataset} מתחלק לשלושה חלקים: ${TERMS_HE.splitTrain} ללמידה, ${TERMS_HE.splitVal} לבחירת הפרומפט, ו${TERMS_HE.splitTest} למדידה סופית`,
  "data.shuffle_explanation": `מערבב את סדר השורות לפני ה${TERMS_HE.split}, כדי שסדר הקובץ לא ישפיע בטעות על התוצאות`,
  "data.split.train": `דוגמאות שה${TERMS_HE.optimizer} משתמש בהן כדי לבנות מועמדים לפרומפט`,
  "data.split.val": `דוגמאות שמדרגות את המועמדים בזמן ה${TERMS_HE.optimization}`,
  "data.split.test": "דוגמאות שמורות למדידה הסופית, אחרי שהפרומפט כבר נבחר",
  "data.seed": `מספר התחלתי קבוע ששומר על אותה חלוקה ואותו ערבוב בכל הרצה חוזרת`,

  "prompt.optimized": `הפרומפט שה${TERMS_HE.optimizer} בנה: הנחיות משופרות ודוגמאות שנבחרו מתוך ה${TERMS_HE.dataset}`,
  "prompt.demonstrations": `דוגמאות קלט-פלט (few-shot demonstrations) שמוצגות ל${TERMS_HE.model} כדי להראות לו את הפורמט והתשובה הרצויים`,

  "module.choice":
    "מודול DSPy הוא רכיב בתוכנית שמפעילה מודל שפה: הוא עוטף כל signature בטכניקת prompting ומגדיר את מבנה הקריאה למודל כדי להפיק את הפלט שמוגדר ב-signature. בתוך המסגרת הזו האופטימייזר מכוון את הפרמטרים הניתנים ללמידה של המודול, כמו הוראות ודוגמאות בפרומפט",
  "module.predict": "Predict — המודול הבסיסי: ממפה את הקלט לפלט בקריאה אחת למודל, ללא שלבי ביניים",
  "module.cot":
    "Chain of Thought — מוסיף שדה reasoning שמוביל את המודל לחשוב שלב-אחר-שלב לפני התשובה הסופית; לרוב משפר דיוק במשימות מורכבות",
  "module.react":
    "ReAct — סוכן שמשלב חשיבה עם קריאה לכלים (tools) בלולאה, עד שהוא מפיק את הפלט שב-signature",
  "module.workflow":
    "Workflow — גרף של כמה צעדים: Signatures, קוד Python וכלים המחוברים זה לזה בקנבס ויזואלי. האופטימיזציה משפרת את ההוראות של כל הצעדים יחד, מול מדד אחד על הפלט הסופי",
  "optimizer.choice": `השיטה שמנסה לשפר את הפרומפט ולמצוא גרסה עם ${TERMS_HE.score} גבוה יותר`,

  "react.tool_source": "מהיכן נטענת רשימת הכלים: שרת MCP חי, או תצלום כלים מתוך הדאטאסט",
  "react.mcp_url": "כתובת שרת ה-MCP שממנו נטענים הכלים של הסוכן",
  "react.auth":
    "כותרת אימות (Authorization header) לשרת ה-MCP. לא נשמרת בשרת ולא נחשפת לסוכן הצ'אט",
  "react.optimized_tools": `הכלים שהסוכן (ReAct) מפעיל בלולאה, עם התיאורים והארגומנטים שה${TERMS_HE.optimizer} חידד במהלך ה${TERMS_HE.optimization}`,

  "config.section.summary": `ה${TERMS_HE.module}, ה${TERMS_HE.optimizer}, והפרמטרים שנבחרו ל${TERMS_HE.optimizationTypeRun} זו`,
  "config.section.models": `מודלי השפה שהוגדרו — ${TERMS_HE.generationModelShort} לייצור תשובות, רפלקציה לניתוח שגיאות`,
  "config.section.data": `חלוקת ה${TERMS_HE.dataset} ל${TERMS_HE.splitTrain}, ${TERMS_HE.splitVal} ו${TERMS_HE.splitTest}, והגדרות ערבוב`,

  "grid.generation_models": `המודלים שמייצרים תשובות. כל ${TERMS_HE.pair} בסריקה משתמש ב${TERMS_HE.generationModel} אחר`,
  "grid.reflection_models": `המודלים שמנתחים שגיאות ומציעים שיפורים. כל ${TERMS_HE.pair} משתמש ב${TERMS_HE.reflectionModel} אחר`,
  "grid.score_comparison": `השוואת ${TERMS_HE.baselineScore} וה${TERMS_HE.optimizedScore} לכל ${TERMS_HE.pair} מודלים`,
  "grid.quality_speed_combined":
    "איכות ומהירות לכל זוג מודלים, זה לצד זה. ככל שהאיכות והמהירות גבוהות יותר, כך הזוג טוב יותר.",
  "grid.avg_response_time_per_pair": "משך זמן ממוצע לכל קריאה למודל שפה, לפי זוג מודלים",
  "grid.best_pair_default": "ברירת מחדל: הזוג עם ציון האיכות הגבוה ביותר. ניתן להחליף לכל זוג אחר.",

  "pair.runtime": `משך ${TERMS_HE.optimizationTypeRun} ה${TERMS_HE.optimization} עבור ${TERMS_HE.pair} המודלים הזה`,

  "serve.section_pair": "כתובת API וקטעי קוד לשילוב הזוג הנבחר באפליקציה שלך",
  "serve.section_run": `כתובת API וקטעי קוד לשילוב ה${TERMS_HE.program} המשופרת באפליקציה שלך`,
  "serve.api_url_pair": "כתובת ה-API של הזוג הנבחר",
  "serve.api_url_run": `כתובת ה-API שאליה שולחים בקשות POST עם שדות הקלט כדי לקבל ${TERMS_HE.prediction} מה${TERMS_HE.program} המשופרת`,
  "serve.api_url_react":
    "כתובת ה-API שאליה שולחים בקשת POST עם הודעת המשתמש; תשובת סוכן ה-ReAct המותאם משודרת בחזרה בזרם SSE",
  "serve.integration_code": "דוגמאות קוד מוכנות להעתקה",

  "submit.depth":
    "כמה רחב החיפוש של GEPA: קל רץ מהר עם פחות ניסיונות; מעמיק בודק יותר אפשרויות ולוקח יותר זמן",
  "submit.reflection_minibatch": `כמה דוגמאות ה${TERMS_HE.model} בודק בכל סבב רפלקציה כדי למצוא דפוסי שגיאה`,
  "submit.eval_rounds": "כמה פעמים להריץ הערכה מלאה כדי לבדוק מועמדים לפרומפט",
  "submit.merge": "כשפעיל, GEPA יכול לבצע merge ולשלב רעיונות מכמה מועמדים טובים לפרומפט אחד",

  "model_config.connection_section": `הרצת ה${TERMS_HE.model} על שרת משלך: נקודת קצה תואמת-OpenAI (Ollama, vLLM, LM Studio או שער ארגוני) ומפתח גישה. השאר/השאירי סגור כדי להשתמש בספקים המובנים`,
  "model_config.model": `ה${TERMS_HE.model} שיריץ את ה${TERMS_HE.optimization}. בחר/י מ${TERMS_HE.modelCatalog}, או מודל מותאם אישית שהתגלה מכתובת ה-Base URL`,
  "model_config.base_url": `כתובת לשרת תואם-OpenAI משלך — Ollama, vLLM, LM Studio או שער ארגוני. השאר/השאירי ריק כדי להשתמש בשרת ברירת המחדל של ה${TERMS_HE.provider}`,
  "model_config.api_key": `מפתח גישה לשרת ה${TERMS_HE.model}. אופציונלי — אם ריק, נלקח ממשתנה סביבה. לא נשמר בשרת ונמחק מהטופס אחרי השליחה`,
  "model_config.temperature": `מידת היצירתיות של ה${TERMS_HE.model} — ערך נמוך נותן תשובות עקביות, גבוה מגוון יותר`,
  "model_config.top_p": `top_p (nucleus sampling): מגביל את מגוון המילים שה${TERMS_HE.model} שוקל — ערך נמוך ממקד, גבוה מאפשר יותר מגוון`,
  "model_config.max_tokens": `אורך ה${TERMS_HE.prediction} המקסימלי — טוקן הוא בערך מילה אחת`,

  "code.signature_metric": `קוד המקור של ה${TERMS_HE.signature} ו${TERMS_HE.metric} שהוגדרו ל${TERMS_HE.optimization} זו`,
  "code.signature": `הגדרת שדות הקלט והפלט של ה${TERMS_HE.task} — מה ה${TERMS_HE.model} מקבל ומה הוא צריך להחזיר`,
  "code.metric": `פונקציה שמודדת את איכות ה${TERMS_HE.prediction} — מחזירה ${TERMS_HE.score} מספרי לכל ${TERMS_HE.example}`,
  "code.predictions_table": `תוצאות הרצת ה${TERMS_HE.program} על דוגמאות הבדיקה — ${TERMS_HE.score} לכל ${TERMS_HE.example} וסיכום כולל`,

  "tagger.upload_file": "העלה/העלי קובץ CSV, JSON או Excel. כל שורה תהפוך לפריט לתיוג",
  "tagger.text_column": "בחר/י את העמודה שמכילה את הטקסט לתיוג. שאר העמודות יישמרו בייצוא",
  "tagger.mode": "בחר/י את סוג התיוג שמתאים למשימה: כן/לא, בחירה מרשימה או טקסט חופשי",
  "tagger.binary_question": "השאלה שתוצג מעל כפתורי כן/לא. כדאי לנסח שאלה שאפשר לענות עליה בבירור",
  "tagger.multiclass_categories":
    "הגדר/הגדירי את הקטגוריות הזמינות לבחירה בזמן התיוג — לפחות שתיים",
  "tagger.freetext_instruction": "ההנחיה שתוצג מעל שדה הטקסט. הסבר/הסבירי בקצרה מה צריך לכתוב",

  "compare.winner_improvement": `אחוז ה${TERMS_HE.scoreImprovement} של ה${TERMS_HE.optimizationTypeRun} הזוכה — ההפרש בין ה${TERMS_HE.optimizedScore} ל${TERMS_HE.baselineScore}`,
  "compare.winner_runtime": `משך הזמן הכולל של ה${TERMS_HE.optimizationTypeRun} הזוכה, מרגע השיגור ועד סיום ה${TERMS_HE.optimization}`,
  "compare.winner_models": `זוג מודלי השפה של ה${TERMS_HE.optimizationTypeRun} הזוכה — ${TERMS_HE.generationModel} שמייצר פלט, ו${TERMS_HE.reflectionModel} שמשפר את ההנחיות`,

  "analytics.score_comparison": `השוואת ${TERMS_HE.baselineScore} מול ה${TERMS_HE.optimizedScore} לכל ${TERMS_HE.optimization} שהושלמה`,
  "analytics.runtime_vs_gain": `ניתוח זמני ${TERMS_HE.optimizationTypeRun} ויעילות — כמה שיפור מתקבל ביחס לזמן`,
  "analytics.runtime_minutes": `משך ה${TERMS_HE.optimizationTypeRun} בדקות לכל ${TERMS_HE.optimization} שהושלמה`,
  "analytics.improvement_per_minute": `אחוזי ${TERMS_HE.scoreImprovement} לכל דקת ${TERMS_HE.optimizationTypeRun} — ערך גבוה משמעו ${TERMS_HE.optimization} יעילה יותר`,
  "analytics.dataset_size_vs_improvement": `האם יותר נתונים מובילים ל${TERMS_HE.scoreImprovement} טוב יותר — כל נקודה היא ${TERMS_HE.optimization} אחת`,
  "analytics.submissions_per_day": `מספר ה${TERMS_HE.optimizationPlural} שהוגשו לפי יום`,
  "analytics.optimizer_avg_improvement": `${TERMS_HE.scoreImprovement} ממוצע באחוזים שכל ${TERMS_HE.optimizer} השיג על פני כל ה${TERMS_HE.optimizationTypeRunPlural}`,
  "analytics.top_improvements": `ה${TERMS_HE.optimizationTypeRunPlural} שהשיגו את השיפור הגדול ביותר בציון, מהטוב לפחות טוב`,
  "analytics.optimizer_comparison_table": `השוואה מפורטת בין ה${TERMS_HE.optimizerPlural}: שיפור ממוצע, מספר ${TERMS_HE.optimizationTypeRunPlural}, וזמן ${TERMS_HE.optimizationTypeRun}`,
  "analytics.model_performance_table": "ביצועי המודלים השונים: תדירות שימוש ושיפור ממוצע",
} as const;

export type TooltipKey = keyof typeof TOOLTIPS;

// Tooltip overlays beyond the Hebrew base (English today), walked via the
// registry fallback chain so a new locale inherits English/Hebrew copy until it
// ships its own overlay.
const TOOLTIP_OVERLAYS: Partial<Record<Locale, Partial<Record<TooltipKey, string>>>> = {
  en: TOOLTIPS_EN,
};

/**
 * Look up tooltip copy by key in the active locale.
 *
 * Walks the active locale's fallback chain over the tooltip overlays and falls
 * back to the Hebrew base, so a key not yet translated for the active locale
 * degrades to its fallback. A missing key everywhere returns the key itself, so
 * the gap surfaces as a dev-visible artifact rather than a blank tooltip.
 */
export function tip(key: TooltipKey): string {
  for (const loc of fallbackChain(getActiveLocale())) {
    const value = TOOLTIP_OVERLAYS[loc]?.[key];
    if (value !== undefined) return value;
  }
  return TOOLTIPS[key] ?? key;
}
