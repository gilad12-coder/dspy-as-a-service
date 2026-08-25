// Split a dspy.Flex module's Python source into its predictor signatures
// (fields + natural-language instructions) and the surrounding code structure.
//
// GEPA rewrites a Flex submodule's whole source on every step, but the change is
// often confined to a predictor's `dspy.Signature("fields", "instructions")` — a
// prompt-level edit wearing a code blob's clothes. Separating the two lets the
// trajectory drawer diff the signature (prose) apart from the code (structure),
// so a pure instructions change reads as one and isn't mistaken for a rewrite.
//
// Best-effort by design: recognises the inline-string signature forms GEPA emits
// (`dspy.Signature("f", "i")`, and the predictor shorthands `dspy.Predict("f",
// "i")` / ChainOfThought / ReAct / RLM / ProgramOfThought). Class-based Signature
// subclasses carry their fields as class attributes, not string args, so they
// yield no signatures and the caller falls back to the whole-module code view.

const ELLIPSIS = "…";

// Predictor constructors whose signature can be given as inline string args
// (`Predict("q -> a", "instructions")`). `Signature` is handled on its own since
// it also appears *nested* inside these — scanning it separately dedupes.
const PREDICTOR_CTORS = [
  "Predict",
  "ChainOfThought",
  "ReAct",
  "RLM",
  "ProgramOfThought",
  "MultiChainComparison",
];

interface StrSpan {
  openIdx: number;
  closeEnd: number;
  contentStart: number;
  contentEnd: number;
  value: string;
}

export interface FlexSignature {
  /** The `self.<name>` the predictor is assigned to, or "" when not found. */
  name: string;
  /** The signature's field spec, e.g. `review: str -> score: int`. */
  fields: string;
  /** The predictor's natural-language instructions (may be empty). */
  instructions: string;
}

export interface FlexDecomposition {
  signatures: FlexSignature[];
  /**
   * The source with every predictor's instructions string elided to `…`. The
   * field spec is left intact — it's genuine structure, so it belongs in the
   * code view — and the instructions render separately as a prompt, so they
   * aren't duplicated here.
   */
  codeSkeleton: string;
}

// Locate every Python string literal in the source, tracking its quote span and
// content bounds. Escape- and comment-aware so quotes inside `#` comments or
// backslash escapes don't miscount; single-quoted literals that run past a line
// break are treated as unterminated and stop at the newline.
function scanStringLiterals(src: string): StrSpan[] {
  const spans: StrSpan[] = [];
  const n = src.length;
  let i = 0;
  while (i < n) {
    const c = src[i];
    if (c === "#") {
      while (i < n && src[i] !== "\n") i += 1;
      continue;
    }
    if (c === '"' || c === "'") {
      const quote = c;
      const triple = src[i + 1] === quote && src[i + 2] === quote;
      const qlen = triple ? 3 : 1;
      const contentStart = i + qlen;
      let j = contentStart;
      while (j < n) {
        const cj = src[j];
        if (cj === "\\") {
          j += 2;
          continue;
        }
        if (triple) {
          if (cj === quote && src[j + 1] === quote && src[j + 2] === quote) break;
          j += 1;
        } else {
          if (cj === quote || cj === "\n") break;
          j += 1;
        }
      }
      const contentEnd = j;
      const closeEnd = triple
        ? j < n
          ? j + 3
          : n
        : j < n && src[j] === quote
          ? j + 1
          : j;
      spans.push({ openIdx: i, closeEnd, contentStart, contentEnd, value: src.slice(contentStart, contentEnd) });
      i = closeEnd;
      continue;
    }
    i += 1;
  }
  return spans;
}

function isInsideString(idx: number, spans: StrSpan[]): boolean {
  return spans.some((s) => idx >= s.openIdx && idx < s.closeEnd);
}

// Find the `)` that closes the call opened at `openIdx`, skipping parens that sit
// inside string literals so quotes-with-parens don't unbalance the count.
function matchCloseParen(src: string, openIdx: number, spans: StrSpan[]): number {
  const n = src.length;
  let depth = 0;
  let i = openIdx;
  while (i < n) {
    const span = spans.find((s) => i >= s.openIdx && i < s.closeEnd);
    if (span !== undefined) {
      i = span.closeEnd;
      continue;
    }
    const c = src[i];
    if (c === "(") depth += 1;
    else if (c === ")") {
      depth -= 1;
      if (depth === 0) return i;
    }
    i += 1;
  }
  return n;
}

// The `self.<name>` of the last attribute assignment before `before` — the
// predictor a (possibly nested) signature is wired to. "" when none precedes it.
function nearestAttrName(src: string, before: number): string {
  const re = /self\.(\w+)\s*=/g;
  let name = "";
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    if (m.index >= before) break;
    name = m[1] ?? "";
  }
  return name;
}

export function decomposeFlexSource(src: string): FlexDecomposition | null {
  const spans = scanStringLiterals(src);
  const stringsInside = (open: number, close: number): StrSpan[] =>
    spans
      .filter((s) => s.openIdx >= open && s.closeEnd <= close)
      .sort((a, b) => a.openIdx - b.openIdx);

  const signatures: FlexSignature[] = [];
  const elide: Array<{ start: number; end: number }> = [];
  const seen = new Set<number>();

  const ctorAlt = ["Signature", ...PREDICTOR_CTORS].join("|");
  const re = new RegExp(`(?:dspy\\.)?\\b(${ctorAlt})\\s*\\(`, "g");
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    const ctor = m[1] ?? "";
    const openParen = m.index + m[0].length - 1;
    if (isInsideString(openParen, spans)) continue;
    const callEnd = matchCloseParen(src, openParen, spans);
    const inside = stringsInside(openParen, callEnd);
    const fields = inside[0];
    if (fields === undefined) continue;
    // A predictor shorthand only counts when its *first* argument is the field
    // string; `Predict(dspy.Signature(...))` has a call there and is left to the
    // Signature scan, so we never double-count the nested signature.
    if (ctor !== "Signature") {
      const between = src.slice(openParen + 1, fields.openIdx);
      if (!/^\s*$/.test(between)) continue;
    }
    if (seen.has(fields.openIdx)) continue;
    seen.add(fields.openIdx);

    // Instructions are the *next positional* string after the fields, reached by
    // a `,` gap — optionally `instructions=` and/or an opening `(` that wraps a
    // multi-line group. Requiring that shape rejects a later string buried in a
    // keyword arg like `tools=["search"]`, which isn't the instructions.
    const chunks: StrSpan[] = [];
    const first = inside[1];
    if (first !== undefined) {
      const gap = src.slice(fields.closeEnd, first.openIdx);
      if (/^\s*,\s*(instructions\s*=\s*)?\(?\s*$/.test(gap)) {
        chunks.push(first);
        // Python implicit string concatenation: adjacent literals separated by
        // only whitespace (`"a" "b"` == `"ab"`). GEPA emits long instructions
        // this way, wrapped in parens across lines — fold them into one string.
        for (let k = 2; k < inside.length; k += 1) {
          const prev = inside[k - 1];
          const next = inside[k];
          if (prev === undefined || next === undefined) break;
          if (!/^\s*$/.test(src.slice(prev.closeEnd, next.openIdx))) break;
          chunks.push(next);
        }
      }
    }

    signatures.push({
      name: nearestAttrName(src, m.index),
      fields: fields.value,
      instructions: chunks.map((s) => s.value).join(""),
    });
    const firstChunk = chunks[0];
    const lastChunk = chunks[chunks.length - 1];
    if (firstChunk !== undefined && lastChunk !== undefined) {
      elide.push({ start: firstChunk.contentStart, end: lastChunk.contentEnd });
    }
  }

  if (signatures.length === 0) return null;

  // Elide right-to-left so each splice leaves earlier offsets untouched.
  let codeSkeleton = src;
  for (const s of [...elide].sort((a, b) => b.start - a.start)) {
    codeSkeleton = codeSkeleton.slice(0, s.start) + ELLIPSIS + codeSkeleton.slice(s.end);
  }
  return { signatures, codeSkeleton };
}

// Pair a candidate predictor with its parent counterpart for the signature diff:
// by the `self.<name>` it's assigned to when that's unambiguous, else by index.
export function matchSignature(
  parent: FlexDecomposition | null,
  sig: FlexSignature,
  idx: number,
): FlexSignature | null {
  if (parent === null) return null;
  if (sig.name.length > 0) {
    const byName = parent.signatures.filter((s) => s.name === sig.name);
    if (byName.length === 1) return byName[0] ?? null;
  }
  return parent.signatures[idx] ?? null;
}
