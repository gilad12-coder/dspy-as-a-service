"use client";

interface SpotlightMaskProps {
  targetRect: DOMRect | null;
  padding?: number;
  borderRadius?: number;
}

export function SpotlightMask({ targetRect, padding = 8, borderRadius = 12 }: SpotlightMaskProps) {
  if (!targetRect) {
    return (
      <svg aria-hidden="true" className="pointer-events-auto absolute inset-0 h-full w-full">
        <rect x="0" y="0" width="100%" height="100%" fill="rgba(28,22,18,0.56)" />
      </svg>
    );
  }

  const edge = 8;
  const x = Math.max(edge, targetRect.x - padding);
  const y = Math.max(edge, targetRect.y - padding);
  const right = Math.min(window.innerWidth - edge, targetRect.right + padding);
  const bottom = Math.min(window.innerHeight - edge, targetRect.bottom + padding);
  const w = Math.max(1, right - x);
  const h = Math.max(1, bottom - y);

  return (
    <svg
      aria-hidden="true"
      className="pointer-events-auto absolute inset-0 h-full w-full"
      data-tutorial-spotlight="true"
    >
      <defs>
        <mask id="tutorial-spotlight-mask">
          <rect x="0" y="0" width="100%" height="100%" fill="white" />
          <rect x={x} y={y} width={w} height={h} rx={borderRadius} fill="black" />
        </mask>
      </defs>

      <rect
        x="0"
        y="0"
        width="100%"
        height="100%"
        fill="rgba(28,22,18,0.56)"
        mask="url(#tutorial-spotlight-mask)"
      />

      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={borderRadius}
        fill="none"
        stroke="rgba(200,168,130,0.95)"
        strokeWidth="2"
        vectorEffect="non-scaling-stroke"
      />

      <rect
        x={x + 2}
        y={y + 2}
        width={Math.max(1, w - 4)}
        height={Math.max(1, h - 4)}
        rx={Math.max(0, borderRadius - 2)}
        fill="none"
        stroke="rgba(250,248,245,0.72)"
        strokeWidth="1"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
