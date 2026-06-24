import { defineConfig, globalIgnores } from "eslint/config";
import i18next from "eslint-plugin-i18next";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

const canonicalTermLiteralSelectors = [
  "Literal[value=/אופטימיזציה|אופטימיזציות|דאטאסט|מודל רפלקציה|מודל מג׳נרט|מודל מג'נרט|אופטימייזר|אופטימייזרים|פונקציית מדידה/u]",
  "TemplateElement[value.raw=/אופטימיזציה|אופטימיזציות|דאטאסט|מודל רפלקציה|מודל מג׳נרט|מודל מג'נרט|אופטימייזר|אופטימייזרים|פונקציית מדידה/u]",
];

const hebrewLiteralSelectors = ["Literal[value=/[א-ת]/u]", "TemplateElement[value.raw=/[א-ת]/u]"];

export default defineConfig([
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: {
      "@typescript-eslint": tseslint.plugin,
    },
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/array-type": ["error", { default: "array-simple" }],
      "@typescript-eslint/consistent-indexed-object-style": ["error", "record"],
      "@typescript-eslint/consistent-type-imports": [
        "error",
        {
          disallowTypeAnnotations: true,
          fixStyle: "separate-type-imports",
          prefer: "type-imports",
        },
      ],
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-inferrable-types": "error",
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: [
                "@/features/*/components/*",
                "@/features/*/hooks/*",
                "@/features/*/lib/*",
                "@/features/*/constants",
              ],
              message:
                "Import feature internals with relative paths inside the same feature, or import from the feature public API.",
            },
          ],
        },
      ],
      "object-shorthand": "error",
      "prefer-template": "error",
    },
  },
  {
    plugins: {
      react,
      "react-hooks": reactHooks,
    },
    settings: {
      react: {
        version: "detect",
      },
    },
    rules: {
      "react-hooks/immutability": "error",
      "react-hooks/preserve-manual-memoization": "error",
      "react-hooks/purity": "error",
      "react-hooks/refs": "error",
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/static-components": "error",
      // Tracked separately; enabling this today creates broad churn in
      // stateful client components without changing runtime behavior.
      "react-hooks/set-state-in-effect": "off",
      "prefer-const": "error",
      "react/no-unescaped-entities": "error",
    },
  },
  {
    files: ["src/**/*.{ts,tsx}"],
    ignores: [
      "src/shared/lib/messages.ts",
      "src/shared/lib/tooltips.ts",
      "src/shared/lib/terms.ts",
      "src/shared/messages/**",
      "src/features/*/messages.ts",
    ],
    plugins: {
      i18next,
    },
    rules: {
      "i18next/no-literal-string": [
        "error",
        {
          framework: "react",
          mode: "jsx-text-only",
          message:
            "Move user-facing JSX text into src/shared/lib/messages.ts or src/shared/lib/terms.ts",
        },
      ],
    },
  },
  {
    files: ["src/**/*.{ts,tsx}"],
    ignores: [
      "src/shared/lib/messages.ts",
      "src/shared/lib/tooltips.ts",
      "src/shared/lib/terms.ts",
      "src/shared/lib/locale.ts",
      "src/shared/lib/generated/**",
      "src/shared/constants/job-status.ts",
      "src/shared/messages/**",
      "src/features/*/messages.ts",
      "src/features/auth/login-samples.ts",
    ],
    rules: {
      "no-restricted-syntax": [
        "error",
        ...hebrewLiteralSelectors.map((selector) => ({
          selector,
          message:
            "Hebrew literals must live in src/shared/lib/messages.ts, tooltips.ts, or terms.ts.",
        })),
        ...canonicalTermLiteralSelectors.map((selector) => ({
          selector,
          message: "Canonical Hebrew domain terms must come from TERMS in src/shared/lib/terms.ts.",
        })),
      ],
    },
  },
  globalIgnores([".next/**", "out/**", "build/**", "next-env.d.ts", "node_modules/**"]),
]);
