"use client";

import { motion, useReducedMotion } from "framer-motion";
import {
  ArrowCounterClockwise,
  ArrowsIn,
  ArrowsOut,
  Crosshair,
  Minus,
  Plus,
} from "@/shared/ui/icons";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { TRAJECTORY_LAYOUT, type LayoutResult } from "../lib/layout";
import { displayCandidateId, type RejectedNode, type TrajectoryNode } from "../lib/types";
import { formatMsg, msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
import { cn } from "@/shared/lib/utils";
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
const DONUT_RING_THICKNESS = 8;
const GHOST_FILL = "#E8E0D3";
const GHOST_STROKE = "rgba(124, 99, 80, 0.5)";
const WINNER_INDICATOR = "#9C7A3F";
const WINNER_HALO = "rgba(156, 122, 63, 0.18)";
const WINNER_FILL = "#F8EBC8";
const WINNER_BADGE_FILL = "#9C7A3F";
const WINNER_BADGE_INK = "#FBF4DF";
// Painted behind score labels (paint-order: stroke) so they stay legible
// where a lineage edge or grid line passes underneath.
const LABEL_HALO = "rgba(250, 248, 245, 0.9)";

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
interface View {
  k: number;
  tx: number;
  ty: number;
}

// Which visual layers of the tree are currently shown; every legend entry
// doubles as the toggle for its layer.
interface LayerVisibility {
  pass: boolean;
  fail: boolean;
  winner: boolean;
  rejected: boolean;
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

// The pass/fail ring is a stroked circle: `pathLength` normalizes the
// circumference to `total` units so each segment spans exactly its fraction,
// and the dash offset rotates the split to start at 12 o'clock (a circle's
// path begins at 3 o'clock). A full ring (all-pass / all-fail) is just an
// undashed stroke — no closed-arc special case needed.
const RING_R = TRAJECTORY_LAYOUT.nodeRadius - DONUT_RING_THICKNESS / 2;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_R;
const RING_SEG_GAP_PX = 1.2;

function ringSegment(
  node: TrajectoryNode,
  key: string,
  fill: string,
  startUnit: number,
  lengthUnits: number,
  total: number,
): React.ReactNode {
  const gapUnits = (RING_SEG_GAP_PX / RING_CIRCUMFERENCE) * total;
  const full = lengthUnits >= total;
  const dashLen = Math.max(0.05, lengthUnits - gapUnits);
  return (
    <circle
      key={key}
      cx={node.x}
      cy={node.y}
      r={RING_R}
      fill="none"
      stroke={fill}
      strokeWidth={DONUT_RING_THICKNESS}
      pathLength={total}
      strokeDasharray={full ? undefined : `${dashLen} ${total - dashLen}`}
      strokeDashoffset={full ? undefined : -(0.75 * total + startUnit)}
    />
  );
}

// Filter layers stay mounted and cross-fade on legend toggles, so switching
// a layer reads as a dissolve rather than a pop. Pointer events are cut
// while hidden so an invisible layer can't swallow clicks.
function LayerFade({
  show,
  reduceMotion,
  children,
}: {
  show: boolean;
  reduceMotion: boolean;
  children: React.ReactNode;
}) {
  return (
    <motion.g
      initial={false}
      animate={{ opacity: show ? 1 : 0 }}
      transition={reduceMotion ? { duration: 0 } : { duration: 0.25, ease: "easeOut" }}
      style={{ pointerEvents: show ? "auto" : "none" }}
    >
      {children}
    </motion.g>
  );
}

function renderRing(
  node: TrajectoryNode,
  showPass: boolean,
  showFail: boolean,
  reduceMotion: boolean,
): React.ReactNode {
  // Always the aggregate pass-fraction split, never per-example ticks — the
  // ring must read identically at every zoom level.
  const passes = node.per_example.filter((e) => e.score > 0).length;
  const total = node.per_example.length;
  return (
    <>
      <LayerFade show={showPass} reduceMotion={reduceMotion}>
        {passes > 0 ? ringSegment(node, "pass", DONUT_PASS_FILL, 0, passes, total) : null}
      </LayerFade>
      <LayerFade show={showFail} reduceMotion={reduceMotion}>
        {passes < total
          ? ringSegment(node, "fail", DONUT_FAIL_FILL, passes, total - passes, total)
          : null}
      </LayerFade>
    </>
  );
}

function fitView(size: { w: number; h: number }, layoutW: number, layoutH: number): View {
  if (size.w < 2 || size.h < 2 || layoutW <= 0 || layoutH <= 0) {
    return { k: 1, tx: 0, ty: 0 };
  }
  const padded = FIT_PADDING_PX * 2;
  const k = clampScale(Math.min((size.w - padded) / layoutW, (size.h - padded) / layoutH));
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
  const [layers, setLayers] = useState<LayerVisibility>({
    pass: true,
    fail: true,
    winner: true,
    rejected: true,
  });
  const toggleLayer = useCallback((key: keyof LayerVisibility) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);
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
            showPass={layers.pass}
            showFail={layers.fail}
            showWinner={layers.winner}
            showRejected={layers.rejected}
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
            <ArrowCounterClockwise className="size-3.5" aria-hidden="true" />
          ) : (
            <Crosshair className="size-3.5" aria-hidden="true" />
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
            <ArrowsIn className="size-3.5" aria-hidden="true" />
          ) : (
            <ArrowsOut className="size-3.5" aria-hidden="true" />
          )}
        </MapControlButton>
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-3 z-10 flex justify-center">
        <div className="pointer-events-auto inline-flex flex-wrap items-center gap-x-1 gap-y-1 rounded-lg border border-[#DDD4C8]/70 bg-background/85 px-2 py-1 text-[11px] font-medium text-muted-foreground/90 backdrop-blur-sm">
          <LegendToggle
            pressed={layers.pass}
            onToggle={() => toggleLayer("pass")}
            swatch={
              <span
                className="inline-block size-2.5 rounded-full"
                style={{ background: DONUT_PASS_FILL }}
              />
            }
            label={msg("trajectory.minibatch.pass_label")}
          />
          <LegendDivider />
          <LegendToggle
            pressed={layers.fail}
            onToggle={() => toggleLayer("fail")}
            swatch={
              <span
                className="inline-block size-2.5 rounded-full"
                style={{ background: DONUT_FAIL_FILL }}
              />
            }
            label={msg("trajectory.minibatch.fail_label")}
          />
          <LegendDivider />
          <LegendToggle
            pressed={layers.winner}
            onToggle={() => toggleLayer("winner")}
            swatch={
              <span
                className="inline-flex items-center justify-center rounded-sm px-1.5 py-0.5 text-[8px] font-semibold leading-none tracking-wider"
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
              <LegendToggle
                pressed={layers.rejected}
                onToggle={() => toggleLayer("rejected")}
                swatch={
                  <span
                    className="inline-block size-2.5 rounded-[2px]"
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
  showPass: boolean;
  showFail: boolean;
  showWinner: boolean;
  showRejected: boolean;
  reduceMotion: boolean;
  onNodeClick: (id: string, e: React.MouseEvent) => void;
  onGhostClick: (rejectionId: string, e: React.MouseEvent) => void;
  onHover: (id: string | null) => void;
}

// The edge/ghost/node geometry is expressed in layout coordinates and never
// depends on the live pan/zoom transform (that lives on the parent <g>). Pulling
// it into a memoized child means a pan or zoom — which fires setView on every
// pointermove/wheel tick — re-renders only the lightweight outer <g>, not the
// hundreds of SVG primitives below it.
const TreeContent = memo(function TreeContent({
  nodes,
  ghosts,
  edges,
  idIndex,
  selectedId,
  hoveredId,
  newestId,
  showPass,
  showFail,
  showWinner,
  showRejected,
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
          // Orthogonal elbow routing: drop from the parent, run flat at the
          // midpoint row, drop into the child. Siblings overlap on the shared
          // vertical stub, which reads as one strict connector bus.
          const midY = (from.y + to.y) / 2;
          const d =
            from.x === to.x
              ? `M ${from.x} ${from.y} L ${to.x} ${to.y}`
              : `M ${from.x} ${from.y} L ${from.x} ${midY} L ${to.x} ${midY} L ${to.x} ${to.y}`;
          return (
            <motion.path
              key={`${edge.from}-${edge.to}-${i}`}
              d={d}
              fill="none"
              stroke={edge.isMerge ? EDGE_STROKE_MERGE : EDGE_STROKE}
              strokeWidth={edge.isMerge ? 1.2 : 1.6}
              strokeDasharray={edge.isMerge ? "4 4" : undefined}
              initial={reduceMotion ? false : { pathLength: 0, opacity: 0 }}
              animate={reduceMotion ? undefined : { pathLength: 1, opacity: 1 }}
              transition={reduceMotion ? undefined : { duration: 0.45, ease: [0.2, 0.8, 0.2, 1] }}
            />
          );
        })}
      </g>

      <LayerFade show={showRejected} reduceMotion={reduceMotion}>
        <g>
          {ghosts.map((ghost) => {
            const parent = idIndex.get(ghost.parent_id);
            if (parent === undefined) return null;
            // Single-corner elbow along the dominant axis, so ghost spokes
            // stay rectilinear like the lineage edges.
            const d =
              Math.abs(ghost.x - parent.x) > Math.abs(ghost.y - parent.y)
                ? `M ${parent.x} ${parent.y} L ${ghost.x} ${parent.y} L ${ghost.x} ${ghost.y}`
                : `M ${parent.x} ${parent.y} L ${parent.x} ${ghost.y} L ${ghost.x} ${ghost.y}`;
            return (
              <path
                key={`ghost-edge-${ghost.rejection_id}`}
                d={d}
                fill="none"
                stroke={EDGE_STROKE_GHOST}
                strokeWidth={1}
                strokeDasharray="3 3"
              />
            );
          })}
          {ghosts.map((ghost) => (
            <motion.rect
              key={`ghost-${ghost.rejection_id}`}
              x={ghost.x - TRAJECTORY_LAYOUT.ghostRadius}
              y={ghost.y - TRAJECTORY_LAYOUT.ghostRadius}
              width={TRAJECTORY_LAYOUT.ghostRadius * 2}
              height={TRAJECTORY_LAYOUT.ghostRadius * 2}
              rx={1.5}
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
      </LayerFade>

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
          const hasRing = node.per_example.length > 0;
          const innerRadius = hasRing
            ? TRAJECTORY_LAYOUT.nodeRadius - DONUT_RING_THICKNESS
            : TRAJECTORY_LAYOUT.nodeRadius;
          const winnerShown = node.isWinner && showWinner;
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
                <LayerFade show={showWinner} reduceMotion={reduceMotion}>
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
                </LayerFade>
              ) : null}
              {hasRing ? renderRing(node, showPass, showFail, reduceMotion) : null}
              <circle
                cx={node.x}
                cy={node.y}
                r={innerRadius}
                fill={winnerShown ? WINNER_FILL : NODE_CORE_FILL}
                stroke={coreStroke}
                strokeWidth={isSelected ? 1.4 : 0.8}
                style={{
                  transition: "fill 250ms ease, stroke 120ms ease, stroke-width 120ms ease",
                }}
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
                <LayerFade show={showWinner} reduceMotion={reduceMotion}>
                  <WinnerBadge x={node.x} y={node.y + TRAJECTORY_LAYOUT.nodeRadius + 4} />
                </LayerFade>
              ) : null}
              <motion.g
                initial={false}
                animate={{ y: node.isWinner && !showWinner ? -20 : 0 }}
                transition={reduceMotion ? { duration: 0 } : { duration: 0.25, ease: "easeOut" }}
              >
                <text
                  x={node.x}
                  y={node.y + TRAJECTORY_LAYOUT.nodeRadius + (node.isWinner ? 34 : 14)}
                  textAnchor="middle"
                  fontFamily="var(--font-mono, monospace)"
                  fontSize="10.5"
                  fontWeight={600}
                  fill="rgba(28, 22, 18, 0.72)"
                  stroke={LABEL_HALO}
                  strokeWidth={2.5}
                  strokeLinejoin="round"
                  pointerEvents="none"
                  style={{ fontVariantNumeric: "tabular-nums", paintOrder: "stroke" }}
                >
                  {node.score.toFixed(2)}
                </text>
              </motion.g>
            </motion.g>
          );
        })}
      </g>
    </>
  );
});

function WinnerBadge({ x, y }: { x: number; y: number }) {
  const label = msg("trajectory.node.winning_label");
  // SVG text can't auto-size its background rect, so estimate the label's
  // width from its glyph count (~5.6px per glyph at 9.5px bold Heebo) and
  // pad each side, so the gold never clips the letters in either locale.
  const w = Math.round(label.length * 5.6) + 14;
  const h = 16;
  const cx = x;
  const top = y + 4;
  return (
    <g pointerEvents="none">
      <rect x={cx - w / 2} y={top} width={w} height={h} rx={3} ry={3} fill={WINNER_BADGE_FILL} />
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

function LegendToggle({
  swatch,
  label,
  pressed,
  onToggle,
}: {
  swatch: React.ReactNode;
  label: string;
  pressed: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onToggle}
      className={cn(
        "inline-flex min-h-[44px] cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-md px-1.5 py-0.5 transition-[opacity,color,background-color] duration-150 hover:bg-accent/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 lg:min-h-0 [@media(hover:none)_and_(pointer:coarse)]:min-h-[44px]",
        !pressed && "opacity-40",
      )}
    >
      {swatch}
      <span>{label}</span>
    </button>
  );
}

function LegendDivider() {
  return <span aria-hidden="true" className="inline-block h-3 w-px bg-border/60" />;
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
              ? "inline-flex size-[44px] items-center justify-center bg-[#1c1612] text-[#faf8f5] transition-[background-color,color] hover:bg-[#2a221c] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#C8A882]/45 lg:size-9 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
              : "inline-flex size-[44px] items-center justify-center text-foreground transition-[background-color,color] hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#C8A882]/45 lg:size-9 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
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
