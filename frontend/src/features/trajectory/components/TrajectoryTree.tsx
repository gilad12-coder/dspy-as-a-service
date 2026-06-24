"use client";

import { motion, useReducedMotion } from "framer-motion";
import {
  LocateFixed,
  Maximize2,
  Minimize2,
  Minus,
  Plus,
  RotateCcw,
} from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { TRAJECTORY_LAYOUT, type LayoutResult } from "../lib/layout";
import { displayCandidateId, type RejectedNode, type TrajectoryNode } from "../lib/types";
import { formatMsg, msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
import {
  Tooltip as UiTooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/shared/ui/primitives/tooltip";

const EDGE_STROKE = "rgba(124, 99, 80, 0.42)";
const EDGE_STROKE_MERGE = "rgba(124, 99, 80, 0.3)";
const EDGE_STROKE_GHOST = "rgba(124, 99, 80, 0.28)";
const NODE_CORE_FILL = "#fdfaf4";
const NODE_CORE_STROKE = "rgba(28, 22, 18, 0.16)";
const NODE_CORE_STROKE_HOVER = "rgba(28, 22, 18, 0.42)";
const NODE_CORE_STROKE_SELECTED = "#1c1612";
const DONUT_PASS_FILL = "#7C8B5A";
const DONUT_FAIL_FILL = "#B26B4A";
const DONUT_SEG_STROKE = "rgba(28, 22, 18, 0.12)";
const DONUT_RING_THICKNESS = 8;
const GHOST_FILL = "#E8E0D3";
const GHOST_STROKE = "rgba(124, 99, 80, 0.5)";
const WINNER_INDICATOR = "#9C7A3F";
const WINNER_HALO = "rgba(156, 122, 63, 0.18)";
const WINNER_FILL = "#F8EBC8";
const WINNER_BADGE_FILL = "#9C7A3F";
const WINNER_BADGE_INK = "#FBF4DF";

const ZOOM_MIN = 0.4;
const ZOOM_MAX = 6;
const ZOOM_WHEEL_FACTOR = 0.0015;
const ZOOM_BUTTON_IN = 1.25;
const ZOOM_BUTTON_OUT = 0.8;
const DRAG_THRESHOLD_PX = 4;
const FIT_PADDING_PX = 32;
const CONTAINER_HEIGHT_PX = 560;
// 44px grid step, oklch grid/axis colors, 48px padding from edges before the
// axes start. Kept local rather than shared to avoid cross-feature coupling
// for two tiny constants.
const GRID_STEP = 44;
const GRID_LINE_COLOR = "oklch(0.91 0.006 50)";
const GRID_AXIS_COLOR = "oklch(0.94 0.005 50)";
const AXIS_PADDING_PX = 48;
// Direct color values — using `hsl(var(--muted))` is invalid because the
// CSS variables hold hex colors, not h s l components, so the gradient gets
// silently dropped and the maximized overlay becomes transparent.
const SURFACE_GRADIENT =
  "radial-gradient(circle at 50% 42%, var(--muted) 0%, var(--background) 58%)";
// Below this zoom level the per-example donut collapses into a single
// pass-fraction ring; tiny segments are unreadable and noisy when far away.
const DONUT_DETAIL_THRESHOLD = 1.1;

interface View {
  k: number;
  tx: number;
  ty: number;
}

export interface TrajectoryTreeProps {
  layout: LayoutResult;
  selectedId: string | null;
  newestId: string | null;
  onSelectCandidate: (id: string) => void;
  onSelectRejected: (rejectionId: string) => void;
  // Optional viewport hint. When provided, the initial fit uses
  // MAX(currentLayout, previewLayout) so the tree opens at the eventual
  // extent — useful in scripted demos where the final size is known up
  // front and we want viewers to see the full graph before nodes stream in.
  previewLayout?: { width: number; height: number };
}

function clampScale(k: number): number {
  return Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, k));
}

function donutSegmentPath(
  cx: number,
  cy: number,
  rOuter: number,
  rInner: number,
  startAngle: number,
  endAngle: number,
): string {
  // A single SVG arc can't draw a closed circle — start and end coordinates
  // coincide and the renderer drops the command, so the all-pass / all-fail
  // donut would disappear when zoomed out (passes === total collapses both
  // segments into one full sweep). Split into two semicircle arcs instead.
  if (endAngle - startAngle >= 2 * Math.PI - 1e-6) {
    return [
      `M ${cx} ${cy - rOuter}`,
      `A ${rOuter} ${rOuter} 0 1 1 ${cx} ${cy + rOuter}`,
      `A ${rOuter} ${rOuter} 0 1 1 ${cx} ${cy - rOuter}`,
      `M ${cx} ${cy - rInner}`,
      `A ${rInner} ${rInner} 0 1 0 ${cx} ${cy + rInner}`,
      `A ${rInner} ${rInner} 0 1 0 ${cx} ${cy - rInner}`,
      "Z",
    ].join(" ");
  }
  const x1 = cx + rOuter * Math.cos(startAngle);
  const y1 = cy + rOuter * Math.sin(startAngle);
  const x2 = cx + rOuter * Math.cos(endAngle);
  const y2 = cy + rOuter * Math.sin(endAngle);
  const x3 = cx + rInner * Math.cos(endAngle);
  const y3 = cy + rInner * Math.sin(endAngle);
  const x4 = cx + rInner * Math.cos(startAngle);
  const y4 = cy + rInner * Math.sin(startAngle);
  const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
  return [
    `M ${x1} ${y1}`,
    `A ${rOuter} ${rOuter} 0 ${largeArc} 1 ${x2} ${y2}`,
    `L ${x3} ${y3}`,
    `A ${rInner} ${rInner} 0 ${largeArc} 0 ${x4} ${y4}`,
    "Z",
  ].join(" ");
}

function renderDonut(node: TrajectoryNode, detailed: boolean): React.ReactNode {
  const cx = node.x;
  const cy = node.y;
  const rOuter = TRAJECTORY_LAYOUT.nodeRadius;
  const rInner = TRAJECTORY_LAYOUT.nodeRadius - DONUT_RING_THICKNESS;
  const passes = node.per_example.filter((e) => e.score > 0).length;
  const total = node.per_example.length;
  if (!detailed && total > 1) {
    const passEnd = -Math.PI / 2 + (passes / total) * 2 * Math.PI;
    const failEnd = -Math.PI / 2 + 2 * Math.PI;
    return (
      <>
        {passes > 0 ? (
          <path
            d={donutSegmentPath(cx, cy, rOuter, rInner, -Math.PI / 2, passEnd)}
            fill={DONUT_PASS_FILL}
            stroke={DONUT_SEG_STROKE}
            strokeWidth={0.5}
          />
        ) : null}
        {passes < total ? (
          <path
            d={donutSegmentPath(cx, cy, rOuter, rInner, passEnd, failEnd)}
            fill={DONUT_FAIL_FILL}
            stroke={DONUT_SEG_STROKE}
            strokeWidth={0.5}
          />
        ) : null}
      </>
    );
  }
  return node.per_example.map((ex, idx) => {
    const start = -Math.PI / 2 + (idx / total) * 2 * Math.PI;
    const end = -Math.PI / 2 + ((idx + 1) / total) * 2 * Math.PI;
    return (
      <path
        key={ex.id}
        d={donutSegmentPath(cx, cy, rOuter, rInner, start, end)}
        fill={ex.score > 0 ? DONUT_PASS_FILL : DONUT_FAIL_FILL}
        stroke={DONUT_SEG_STROKE}
        strokeWidth={0.5}
      />
    );
  });
}

function fitView(size: { w: number; h: number }, layoutW: number, layoutH: number): View {
  if (size.w < 2 || size.h < 2 || layoutW <= 0 || layoutH <= 0) {
    return { k: 1, tx: 0, ty: 0 };
  }
  const padded = FIT_PADDING_PX * 2;
  const k = clampScale(
    Math.min((size.w - padded) / layoutW, (size.h - padded) / layoutH),
  );
  const tx = (size.w - layoutW * k) / 2;
  const ty = (size.h - layoutH * k) / 2;
  return { k, tx, ty };
}

export function TrajectoryTree({
  layout,
  selectedId,
  newestId,
  onSelectCandidate,
  onSelectRejected,
  previewLayout,
}: TrajectoryTreeProps) {
  const reduceMotion = useReducedMotion();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<View>({ k: 1, tx: 0, ty: 0 });
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const panStateRef = useRef<null | {
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startTx: number;
    startTy: number;
    moved: boolean;
  }>(null);
  // Auto-fit keeps re-centering as new candidates stream in until the user
  // actively pans, wheel-zooms, or button-zooms — once they take control we
  // freeze their framing. Reset / maximize-toggle release the lock.
  const userInteractedRef = useRef(false);

  const { nodes, ghosts, edges, width, height } = layout;
  const fitWidth = Math.max(width, previewLayout?.width ?? 0);
  const fitHeight = Math.max(height, previewLayout?.height ?? 0);
  const idIndex = useMemo(() => {
    const m = new Map<string, TrajectoryNode>();
    for (const n of nodes) m.set(n.candidate_id, n);
    return m;
  }, [nodes]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    // Seed size synchronously so the SVG viewBox is correct on the first paint
    // after a portal move (ResizeObserver's initial dispatch is async and would
    // otherwise paint one frame with the previous container's dimensions).
    const r = el.getBoundingClientRect();
    setSize({ w: r.width, h: r.height });
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width: w, height: h } = e.contentRect;
        setSize({ w, h });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
    // Re-attach when the maximize toggle moves the container via portal: React
    // unmounts/remounts the host div, so the previous observer is stale.
  }, [isMaximized]);

  useEffect(() => {
    if (size.w < 2 || size.h < 2) return;
    if (userInteractedRef.current) return;
    setView(fitView(size, fitWidth, fitHeight));
  }, [size, fitWidth, fitHeight]);

  // Toggling maximize resizes the container; release the interaction lock so
  // the new viewport gets a fresh fit.
  useEffect(() => {
    userInteractedRef.current = false;
  }, [isMaximized]);

  // Lock page scroll while the maximized overlay is open so the user cannot
  // scroll the underlying content behind the fixed surface, and bind ESC to
  // exit so the overlay behaves like a normal modal.
  useEffect(() => {
    if (!isMaximized) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setIsMaximized(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [isMaximized]);

  const zoomAt = useCallback((cx: number, cy: number, factor: number) => {
    setView((v) => {
      const nextK = clampScale(v.k * factor);
      if (nextK === v.k) return v;
      userInteractedRef.current = true;
      const wx = (cx - v.tx) / v.k;
      const wy = (cy - v.ty) / v.k;
      return { k: nextK, tx: cx - wx * nextK, ty: cy - wy * nextK };
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const rect = container.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * ZOOM_WHEEL_FACTOR);
      zoomAt(e.clientX - rect.left, e.clientY - rect.top, factor);
    };
    container.addEventListener("wheel", onWheel, { passive: false });
    return () => container.removeEventListener("wheel", onWheel);
    // Re-attach when maximize portals the container: the old element is
    // detached and a new one mounts in the portal target, leaving the
    // previous listener orphaned (same reason the ResizeObserver re-binds).
  }, [zoomAt, isMaximized]);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (e.pointerType === "mouse" && e.button !== 0) return;
      // Don't hijack pointers that land on the floating zoom controls.
      const target = e.target as Element | null;
      if (target?.closest("[data-trajectory-controls]")) return;
      // Capturing on pointerdown would steal the synthesized click from child
      // nodes — defer until the user actually starts panning (see pointermove).
      panStateRef.current = {
        pointerId: e.pointerId,
        startClientX: e.clientX,
        startClientY: e.clientY,
        startTx: view.tx,
        startTy: view.ty,
        moved: false,
      };
    },
    [view.tx, view.ty],
  );

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const ps = panStateRef.current;
    if (!ps || ps.pointerId !== e.pointerId) return;
    const dx = e.clientX - ps.startClientX;
    const dy = e.clientY - ps.startClientY;
    if (!ps.moved && Math.hypot(dx, dy) > DRAG_THRESHOLD_PX) {
      ps.moved = true;
      userInteractedRef.current = true;
      setIsDragging(true);
      (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    }
    if (ps.moved) {
      setView((v) => ({ k: v.k, tx: ps.startTx + dx, ty: ps.startTy + dy }));
    }
  }, []);

  const handlePointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const ps = panStateRef.current;
    if (ps && ps.pointerId === e.pointerId) {
      panStateRef.current = null;
      if (ps.moved) {
        (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
        setIsDragging(false);
      }
    }
  }, []);

  const handleNodeClick = useCallback(
    (id: string, e: React.MouseEvent) => {
      // Suppress clicks that close a pan gesture — drag-then-release should
      // not also select a node it happened to release on.
      if (panStateRef.current?.moved) return;
      e.stopPropagation();
      onSelectCandidate(id);
    },
    [onSelectCandidate],
  );

  const handleGhostClick = useCallback(
    (rejectionId: string, e: React.MouseEvent) => {
      if (panStateRef.current?.moved) return;
      e.stopPropagation();
      onSelectRejected(rejectionId);
    },
    [onSelectRejected],
  );

  const zoomFromCenter = useCallback(
    (factor: number) => zoomAt(size.w / 2, size.h / 2, factor),
    [size.w, size.h, zoomAt],
  );
  const resetView = useCallback(() => {
    userInteractedRef.current = false;
    setView(fitView(size, fitWidth, fitHeight));
  }, [size, fitWidth, fitHeight]);
  const isTransformed = useMemo(() => {
    const baseline = fitView(size, fitWidth, fitHeight);
    return view.k !== baseline.k || view.tx !== baseline.tx || view.ty !== baseline.ty;
  }, [view, size, fitWidth, fitHeight]);

  if (nodes.length === 0) return null;

  const transform = `translate(${view.tx}, ${view.ty}) scale(${view.k})`;

  const gridScreenStep = GRID_STEP * view.k;
  const axisCx = (size.w / 2) * view.k + view.tx;
  const axisCy = (size.h / 2) * view.k + view.ty;

  const treeBody = (
    <div
      ref={containerRef}
      className={
        isMaximized
          ? "fixed inset-0 z-50 w-screen h-screen overflow-hidden border-0"
          : "relative w-full overflow-hidden rounded-xl border border-[#DDD4C8]/60"
      }
      style={
        isMaximized
          ? { background: SURFACE_GRADIENT }
          : { height: CONTAINER_HEIGHT_PX, background: SURFACE_GRADIENT }
      }
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      role="tree"
      aria-label={msg("trajectory.a11y.tree_label")}
    >
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${Math.max(size.w, 1)} ${Math.max(size.h, 1)}`}
        preserveAspectRatio="xMidYMid meet"
        style={{
          display: "block",
          direction: "ltr",
          cursor: isDragging ? "grabbing" : "grab",
          touchAction: "none",
        }}
      >
        {gridScreenStep > 2 ? (
          <>
            <defs>
              <pattern
                id="trajectory-grid"
                x={view.tx}
                y={view.ty}
                width={gridScreenStep}
                height={gridScreenStep}
                patternUnits="userSpaceOnUse"
              >
                <path
                  d={`M ${gridScreenStep} 0 L 0 0 L 0 ${gridScreenStep}`}
                  fill="none"
                  stroke={GRID_LINE_COLOR}
                  strokeWidth={1}
                />
              </pattern>
            </defs>
            <rect width={size.w} height={size.h} fill="url(#trajectory-grid)" />
            <line
              x1={axisCx}
              y1={AXIS_PADDING_PX}
              x2={axisCx}
              y2={Math.max(size.h - AXIS_PADDING_PX, AXIS_PADDING_PX)}
              stroke={GRID_AXIS_COLOR}
              strokeWidth={1}
            />
            <line
              x1={AXIS_PADDING_PX}
              y1={axisCy}
              x2={Math.max(size.w - AXIS_PADDING_PX, AXIS_PADDING_PX)}
              y2={axisCy}
              stroke={GRID_AXIS_COLOR}
              strokeWidth={1}
            />
          </>
        ) : null}
        <g transform={transform}>
          <TreeContent
            nodes={nodes}
            ghosts={ghosts}
            edges={edges}
            idIndex={idIndex}
            selectedId={selectedId}
            hoveredId={hoveredId}
            newestId={newestId}
            detailed={view.k >= DONUT_DETAIL_THRESHOLD}
            reduceMotion={!!reduceMotion}
            onNodeClick={handleNodeClick}
            onGhostClick={handleGhostClick}
            onHover={setHoveredId}
          />
        </g>
      </svg>

      <div
        data-trajectory-controls
        className={
          isMaximized
            ? "absolute top-4 start-4 z-20 flex overflow-hidden rounded-lg border border-border/70 bg-background/95 shadow-md backdrop-blur-sm"
            : "absolute top-3 end-3 z-20 flex overflow-hidden rounded-lg border border-border/70 bg-background/90 shadow-sm backdrop-blur-sm"
        }
      >
        <MapControlButton
          label={msg("trajectory.controls.zoom_in")}
          onClick={() => zoomFromCenter(ZOOM_BUTTON_IN)}
        >
          <Plus className="size-3.5" aria-hidden="true" />
        </MapControlButton>
        <MapControlButton
          label={msg("trajectory.controls.zoom_out")}
          onClick={() => zoomFromCenter(ZOOM_BUTTON_OUT)}
        >
          <Minus className="size-3.5" aria-hidden="true" />
        </MapControlButton>
        <MapControlButton label={msg("trajectory.controls.zoom_reset")} onClick={resetView}>
          {isTransformed ? (
            <RotateCcw className="size-3.5" aria-hidden="true" />
          ) : (
            <LocateFixed className="size-3.5" aria-hidden="true" />
          )}
        </MapControlButton>
        <ControlsDivider />
        <MapControlButton
          label={
            isMaximized
              ? msg("trajectory.controls.fullscreen_exit")
              : msg("trajectory.controls.fullscreen_enter")
          }
          onClick={() => setIsMaximized((v) => !v)}
        >
          {isMaximized ? (
            <Minimize2 className="size-3.5" aria-hidden="true" />
          ) : (
            <Maximize2 className="size-3.5" aria-hidden="true" />
          )}
        </MapControlButton>
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-3 z-10 flex justify-center">
        <div
          className="pointer-events-auto inline-flex flex-wrap items-center gap-x-3 gap-y-1 rounded-full border border-[#DDD4C8]/70 bg-background/85 px-4 py-1.5 text-[11px] font-medium text-muted-foreground/90 backdrop-blur-sm"
        >
          <LegendItem
            swatch={
              <span
                className="inline-block size-2.5 rounded-full"
                style={{ background: DONUT_PASS_FILL }}
              />
            }
            label={msg("trajectory.minibatch.pass_label")}
          />
          <LegendDivider />
          <LegendItem
            swatch={
              <span
                className="inline-block size-2.5 rounded-full"
                style={{ background: DONUT_FAIL_FILL }}
              />
            }
            label={msg("trajectory.minibatch.fail_label")}
          />
          <LegendDivider />
          <LegendItem
            swatch={
              <span
                className="inline-flex items-center justify-center rounded-full px-1.5 py-0.5 text-[8px] font-semibold leading-none tracking-wider"
                style={{
                  background: WINNER_BADGE_FILL,
                  color: WINNER_BADGE_INK,
                }}
              >
                {msg("trajectory.node.winning_label")}
              </span>
            }
            label={TERMS.winningCandidate}
          />
          {ghosts.length > 0 ? (
            <>
              <LegendDivider />
              <LegendItem
                swatch={
                  <span
                    className="inline-block size-2.5 rounded-full"
                    style={{
                      background: GHOST_FILL,
                      border: `1px solid ${GHOST_STROKE}`,
                    }}
                  />
                }
                label={msg("trajectory.ghost.legend")}
              />
            </>
          ) : null}
        </div>
      </div>
    </div>
  );

  if (isMaximized && typeof document !== "undefined") {
    return (
      <>
        {/* Placeholder keeps the panel's vertical rhythm intact while the
            tree is portaled into a viewport-spanning overlay above. */}
        <div
          aria-hidden="true"
          className="w-full rounded-xl border border-[#DDD4C8]/60 opacity-40"
          style={{ height: CONTAINER_HEIGHT_PX, background: SURFACE_GRADIENT }}
        />
        {createPortal(treeBody, document.body)}
      </>
    );
  }
  return treeBody;
}

interface TreeContentProps {
  nodes: TrajectoryNode[];
  ghosts: RejectedNode[];
  edges: LayoutResult["edges"];
  idIndex: Map<string, TrajectoryNode>;
  selectedId: string | null;
  hoveredId: string | null;
  newestId: string | null;
  detailed: boolean;
  reduceMotion: boolean;
  onNodeClick: (id: string, e: React.MouseEvent) => void;
  onGhostClick: (rejectionId: string, e: React.MouseEvent) => void;
  onHover: (id: string | null) => void;
}

// The edge/ghost/node geometry is expressed in layout coordinates and never
// depends on the live pan/zoom transform (that lives on the parent <g>). Pulling
// it into a memoized child means a pan or zoom — which fires setView on every
// pointermove/wheel tick — re-renders only the lightweight outer <g>, not the
// hundreds of SVG primitives below it. `detailed` (the donut-collapse boolean)
// is the one zoom-derived input, passed as a bool so crossing the threshold is
// the only zoom event that reconciles the tree.
const TreeContent = memo(function TreeContent({
  nodes,
  ghosts,
  edges,
  idIndex,
  selectedId,
  hoveredId,
  newestId,
  detailed,
  reduceMotion,
  onNodeClick,
  onGhostClick,
  onHover,
}: TreeContentProps) {
  return (
    <>
      <g>
        {edges.map((edge, i) => {
          const from = idIndex.get(edge.from);
          const to = idIndex.get(edge.to);
          if (from === undefined || to === undefined) return null;
          return (
            <motion.line
              key={`${edge.from}-${edge.to}-${i}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={edge.isMerge ? EDGE_STROKE_MERGE : EDGE_STROKE}
              strokeWidth={edge.isMerge ? 1.2 : 1.6}
              strokeLinecap="round"
              strokeDasharray={edge.isMerge ? "4 4" : undefined}
              initial={reduceMotion ? false : { pathLength: 0, opacity: 0 }}
              animate={reduceMotion ? undefined : { pathLength: 1, opacity: 1 }}
              transition={reduceMotion ? undefined : { duration: 0.45, ease: [0.2, 0.8, 0.2, 1] }}
            />
          );
        })}
      </g>

      <g>
        {ghosts.map((ghost) => {
          const parent = idIndex.get(ghost.parent_id);
          if (parent === undefined) return null;
          return (
            <line
              key={`ghost-edge-${ghost.rejection_id}`}
              x1={parent.x}
              y1={parent.y}
              x2={ghost.x}
              y2={ghost.y}
              stroke={EDGE_STROKE_GHOST}
              strokeWidth={1}
              strokeDasharray="3 3"
            />
          );
        })}
        {ghosts.map((ghost) => (
          <motion.circle
            key={`ghost-${ghost.rejection_id}`}
            cx={ghost.x}
            cy={ghost.y}
            r={TRAJECTORY_LAYOUT.ghostRadius}
            fill={GHOST_FILL}
            stroke={GHOST_STROKE}
            strokeWidth={0.9}
            initial={reduceMotion ? false : { scale: 0.5, opacity: 0 }}
            animate={reduceMotion ? undefined : { scale: 1, opacity: 1 }}
            transition={reduceMotion ? undefined : { duration: 0.35, ease: [0.2, 0.8, 0.2, 1] }}
            style={{ cursor: "pointer" }}
            onClick={(e) => onGhostClick(ghost.rejection_id, e)}
          />
        ))}
      </g>

      <g>
        {nodes.map((node) => {
          const isSelected = node.candidate_id === selectedId;
          const isHovered = node.candidate_id === hoveredId;
          const isNewest = node.candidate_id === newestId;
          const coreStroke = isSelected
            ? NODE_CORE_STROKE_SELECTED
            : isHovered
              ? NODE_CORE_STROKE_HOVER
              : NODE_CORE_STROKE;
          const hasDonut = node.per_example.length > 0;
          const innerRadius = hasDonut
            ? TRAJECTORY_LAYOUT.nodeRadius - DONUT_RING_THICKNESS
            : TRAJECTORY_LAYOUT.nodeRadius;
          return (
            <motion.g
              key={node.candidate_id}
              role="treeitem"
              aria-label={formatMsg("trajectory.a11y.node_label", {
                id: displayCandidateId(node.candidate_id),
                gen: node.generation,
                score: node.score.toFixed(2),
              })}
              aria-selected={isSelected}
              tabIndex={isSelected ? 0 : -1}
              onMouseEnter={() => onHover(node.candidate_id)}
              onMouseLeave={() => onHover(null)}
              onClick={(e) => onNodeClick(node.candidate_id, e)}
              initial={
                reduceMotion
                  ? false
                  : isNewest
                    ? { scale: 0, opacity: 0 }
                    : { scale: 0.7, opacity: 0 }
              }
              animate={reduceMotion ? undefined : { scale: 1, opacity: 1 }}
              transition={reduceMotion ? undefined : { duration: 0.45, ease: [0.2, 0.8, 0.2, 1] }}
              style={{ cursor: "pointer" }}
            >
              {isNewest && !reduceMotion ? (
                <motion.circle
                  cx={node.x}
                  cy={node.y}
                  r={TRAJECTORY_LAYOUT.nodeRadius}
                  fill="none"
                  stroke={WINNER_INDICATOR}
                  strokeWidth="1.6"
                  initial={{ r: TRAJECTORY_LAYOUT.nodeRadius, opacity: 0.6 }}
                  animate={{ r: TRAJECTORY_LAYOUT.nodeRadius * 2.2, opacity: 0 }}
                  transition={{ duration: 1.2, ease: "easeOut", repeat: 2 }}
                />
              ) : null}
              {isSelected ? (
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={TRAJECTORY_LAYOUT.nodeRadius + 5}
                  fill="none"
                  stroke={NODE_CORE_STROKE_SELECTED}
                  strokeWidth={1.5}
                  strokeOpacity={0.7}
                />
              ) : null}
              {node.isWinner ? (
                <>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={TRAJECTORY_LAYOUT.nodeRadius + 11}
                    fill="none"
                    stroke={WINNER_HALO}
                    strokeWidth={3}
                  />
                  {reduceMotion ? null : (
                    <motion.circle
                      cx={node.x}
                      cy={node.y}
                      r={TRAJECTORY_LAYOUT.nodeRadius + 4}
                      fill="none"
                      stroke={WINNER_INDICATOR}
                      strokeWidth={1.4}
                      initial={{
                        opacity: 0.5,
                        r: TRAJECTORY_LAYOUT.nodeRadius + 4,
                      }}
                      animate={{
                        opacity: [0.5, 0.12, 0.5],
                        r: [
                          TRAJECTORY_LAYOUT.nodeRadius + 4,
                          TRAJECTORY_LAYOUT.nodeRadius + 10,
                          TRAJECTORY_LAYOUT.nodeRadius + 4,
                        ],
                      }}
                      transition={{
                        duration: 3.2,
                        ease: "easeInOut",
                        repeat: Infinity,
                      }}
                    />
                  )}
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={TRAJECTORY_LAYOUT.nodeRadius + 4}
                    fill="none"
                    stroke={WINNER_INDICATOR}
                    strokeWidth={2.6}
                  />
                </>
              ) : null}
              {hasDonut ? renderDonut(node, detailed) : null}
              <circle
                cx={node.x}
                cy={node.y}
                r={innerRadius}
                fill={node.isWinner ? WINNER_FILL : NODE_CORE_FILL}
                stroke={coreStroke}
                strokeWidth={isSelected ? 1.4 : 0.8}
              />
              <text
                x={node.x}
                y={node.y + 5}
                textAnchor="middle"
                fontFamily="var(--font-mono, monospace)"
                fontSize="14"
                fontWeight={700}
                fill="#1c1612"
                pointerEvents="none"
              >
                {displayCandidateId(node.candidate_id)}
              </text>
              {node.isWinner ? (
                <WinnerBadge x={node.x} y={node.y + TRAJECTORY_LAYOUT.nodeRadius + 4} />
              ) : null}
              <text
                x={node.x}
                y={node.y + TRAJECTORY_LAYOUT.nodeRadius + (node.isWinner ? 32 : 14)}
                textAnchor="middle"
                fontFamily="var(--font-mono, monospace)"
                fontSize="10.5"
                fontWeight={600}
                fill="rgba(28, 22, 18, 0.72)"
                pointerEvents="none"
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {node.score.toFixed(2)}
              </text>
            </motion.g>
          );
        })}
      </g>
    </>
  );
});

function WinnerBadge({ x, y }: { x: number; y: number }) {
  const label = msg("trajectory.node.winning_label");
  const w = 36;
  const h = 15;
  const cx = x;
  const top = y + 4;
  return (
    <g pointerEvents="none">
      <rect
        x={cx - w / 2}
        y={top}
        width={w}
        height={h}
        rx={h / 2}
        ry={h / 2}
        fill={WINNER_BADGE_FILL}
      />
      <text
        x={cx}
        y={top + h / 2 + 3.4}
        textAnchor="middle"
        fontFamily='"Heebo", "Assistant", system-ui, sans-serif'
        fontSize="9.5"
        fontWeight={700}
        letterSpacing="0.4"
        fill={WINNER_BADGE_INK}
      >
        {label}
      </text>
    </g>
  );
}

function LegendItem({ swatch, label }: { swatch: React.ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      {swatch}
      <span>{label}</span>
    </span>
  );
}

function LegendDivider() {
  return (
    <span
      aria-hidden="true"
      className="inline-block h-3 w-px bg-border/60"
    />
  );
}

function MapControlButton({
  label,
  onClick,
  pressed,
  children,
}: {
  label: string;
  onClick: () => void;
  pressed?: boolean;
  children: React.ReactNode;
}) {
  return (
    <UiTooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          aria-label={label}
          aria-pressed={pressed}
          className={
            pressed === true
              ? "inline-flex size-9 items-center justify-center bg-[#1c1612] text-[#faf8f5] transition-[background-color,color] hover:bg-[#2a221c] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#C8A882]/45"
              : "inline-flex size-9 items-center justify-center text-foreground transition-[background-color,color] hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#C8A882]/45"
          }
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={8}>
        {label}
      </TooltipContent>
    </UiTooltip>
  );
}

function ControlsDivider() {
  return (
    <span
      aria-hidden="true"
      className="my-1.5 inline-block w-px bg-border/60"
      style={{ alignSelf: "stretch" }}
    />
  );
}

