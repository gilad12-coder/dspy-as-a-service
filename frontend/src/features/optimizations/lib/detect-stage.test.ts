import { test } from "node:test";
import assert from "node:assert/strict";
import type { GridSearchResult, OptimizationStatusResponse, ProgressEvent } from "@/shared/types/api";
import { detectPairStage, detectStage } from "./detect-stage.ts";

const ev = (event: string, metrics: Record<string, unknown> = {}): ProgressEvent => ({
  timestamp: "",
  event,
  metrics,
});

const job = (o: Partial<OptimizationStatusResponse>): OptimizationStatusResponse =>
  o as OptimizationStatusResponse;

test("terminal job states short-circuit stage detection", () => {
  assert.equal(detectStage(job({ status: "validating" })), "validating");
  assert.equal(detectStage(job({ status: "success" })), "done");
});

test("progress events drive the running-job stage", () => {
  const at = (name: string) => detectStage(job({ status: "running", progress_events: [ev(name)] }));
  assert.equal(at("optimized_evaluated"), "done");
  assert.equal(at("optimizer_progress"), "optimizing");
  assert.equal(at("baseline_evaluated"), "optimizing");
  assert.equal(at("candidate"), "optimizing");
  assert.equal(at("grid_pair_started"), "baseline");
  assert.equal(at("dataset_splits_ready"), "baseline");
});

test("a running job with no events falls back to tqdm hints, then to splitting", () => {
  assert.equal(detectStage(job({ status: "running", latest_metrics: { tqdm_percent: 40 } })), "optimizing");
  assert.equal(detectStage(job({ status: "running" })), "splitting");
});

test("detectPairStage reads a pair's own final result before the event stream", () => {
  const grid = { pair_results: [{ pair_index: 1, optimized_test_metric: 0.82 }] } as unknown as GridSearchResult;
  assert.equal(detectPairStage(job({ status: "running", progress_events: [], grid_result: grid }), 1), "done");
});

test("detectPairStage classifies by the matching pair's events", () => {
  const completed = [ev("grid_pair_started", { pair_index: 0 }), ev("grid_pair_completed", { pair_index: 0 })];
  assert.equal(detectPairStage(job({ status: "running", progress_events: completed }), 0), "done");
  assert.equal(
    detectPairStage(job({ status: "running", progress_events: [ev("grid_pair_started", { pair_index: 0 })] }), 0),
    "baseline",
  );
});
