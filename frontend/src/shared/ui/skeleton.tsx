"use client";

import "react-loading-skeleton/dist/skeleton.css";

import Skeleton, { SkeletonTheme } from "react-loading-skeleton";
import type { ReactNode } from "react";

export { Skeleton };

const BASE_COLOR = "#ebe4d8";
const HIGHLIGHT_COLOR = "#f7f1e6";
const DURATION_SECONDS = 1.4;
const BORDER_RADIUS = 6;

interface AppSkeletonThemeProps {
  children: ReactNode;
}

export function AppSkeletonTheme({ children }: AppSkeletonThemeProps) {
  return (
    <SkeletonTheme
      baseColor={BASE_COLOR}
      highlightColor={HIGHLIGHT_COLOR}
      duration={DURATION_SECONDS}
      borderRadius={BORDER_RADIUS}
    >
      {children}
    </SkeletonTheme>
  );
}
