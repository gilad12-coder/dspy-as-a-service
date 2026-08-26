export function isReactModuleName(value: string | null | undefined): boolean {
  return value?.trim().toLowerCase() === "react";
}
