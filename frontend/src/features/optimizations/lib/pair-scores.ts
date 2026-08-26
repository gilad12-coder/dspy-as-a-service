import type { PairResult } from "@/shared/types/api";

interface PairScores {
  pair_index: number;
  quality: number;
  speed: number | null;
}

export interface GridScoring {
  byIndex: Record<number, PairScores>;
  qualityWinner: number | null;
  speedWinner: number | null;
  overallWinner: number | null;
  hasSpeedData: boolean;
}

// Metrics are stored on the canonical 0–100 scale; fold to 0–1 so quality
// and speed share a unit range for the combined ranking and scatter charts.
function normalizeQuality(v: number | undefined | null): number {
  if (v == null) return 0;
  return Math.max(0, Math.min(1, v / 100));
}

// Speed score: fastest pair = 1.0, others scale as min_time / their_time.
// Pairs without latency data get speed = null so they can't win speed.
export function computePairScores(pairs: PairResult[]): GridScoring {
  const successful = pairs.filter((p) => !p.error && p.optimized_test_metric != null);
  if (successful.length === 0) {
    return {
      byIndex: {},
      qualityWinner: null,
      speedWinner: null,
      overallWinner: null,
      hasSpeedData: false,
    };
  }

  const times = successful
    .map((p) => p.avg_response_time_ms)
    .filter((t): t is number => t != null && t > 0);
  const minTime = times.length ? Math.min(...times) : null;
  const hasSpeedData = minTime != null;

  const byIndex: Record<number, PairScores> = {};
  for (const p of successful) {
    const quality = normalizeQuality(p.optimized_test_metric);
    const t = p.avg_response_time_ms;
    const speed = minTime != null && t != null && t > 0 ? minTime / t : null;
    byIndex[p.pair_index] = { pair_index: p.pair_index, quality, speed };
  }

  const pickMax = (key: "quality" | "speed"): number | null => {
    let best: PairScores | null = null;
    for (const s of Object.values(byIndex)) {
      const v = s[key];
      if (v == null) continue;
      if (best == null || (s[key] as number) > (best[key] as number)) best = s;
    }
    return best?.pair_index ?? null;
  };

  const qualityWinner = pickMax("quality");
  const speedWinner = pickMax("speed");

  return {
    byIndex,
    qualityWinner,
    speedWinner,
    overallWinner: qualityWinner,
    hasSpeedData,
  };
}
