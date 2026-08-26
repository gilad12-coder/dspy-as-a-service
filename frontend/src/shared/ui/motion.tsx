"use client";

import { motion, useReducedMotion, type HTMLMotionProps } from "framer-motion";
import * as React from "react";
import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";

export const FadeIn = React.memo(function FadeIn({
  children,
  delay = 0,
  direction = "up",
  className,
  ...props
}: {
  children: React.ReactNode;
  delay?: number;
  direction?: "up" | "down" | "left" | "right";
  className?: string;
} & Omit<HTMLMotionProps<"div">, "initial" | "whileInView" | "viewport" | "transition">) {
  const shouldReduceMotion = useReducedMotion();
  const offsets = {
    up: { y: 24 },
    down: { y: -24 },
    left: { x: 24 },
    right: { x: -24 },
  };

  return (
    <motion.div
      initial={shouldReduceMotion ? false : { opacity: 0, ...offsets[direction] }}
      whileInView={shouldReduceMotion ? undefined : { opacity: 1, x: 0, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={
        shouldReduceMotion ? undefined : { duration: 0.5, delay, ease: [0.2, 0.8, 0.2, 1] }
      }
      className={className}
      {...props}
    >
      {children}
    </motion.div>
  );
});

export const StaggerContainer = React.memo(function StaggerContainer({
  children,
  className,
  staggerDelay = 0.08,
}: {
  children: React.ReactNode;
  className?: string;
  staggerDelay?: number;
}) {
  const shouldReduceMotion = useReducedMotion();
  return (
    <motion.div
      initial={shouldReduceMotion ? false : "hidden"}
      whileInView={shouldReduceMotion ? undefined : "visible"}
      viewport={{ once: true, margin: "-40px" }}
      variants={{
        hidden: {},
        visible: { transition: { staggerChildren: staggerDelay } },
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
});

export const StaggerItem = React.memo(function StaggerItem({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const shouldReduceMotion = useReducedMotion();
  return (
    <motion.div
      variants={
        shouldReduceMotion
          ? undefined
          : {
              hidden: { opacity: 0, y: 20, scale: 0.97 },
              visible: {
                opacity: 1,
                y: 0,
                scale: 1,
                transition: { duration: 0.45, ease: [0.2, 0.8, 0.2, 1] },
              },
            }
      }
      className={className}
    >
      {children}
    </motion.div>
  );
});

export function TiltCard({
  children,
  className,
  onClick,
}: {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}) {
  const ref = React.useRef<HTMLDivElement>(null);
  const shouldReduceMotion = useReducedMotion();

  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (shouldReduceMotion) return;
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      el.style.transform = `perspective(800px) rotateY(${x * 6}deg) rotateX(${-y * 6}deg) translateY(-2px)`;
    },
    [shouldReduceMotion],
  );

  const handleMouseLeave = React.useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.transform = "perspective(800px) rotateY(0deg) rotateX(0deg) translateY(0)";
  }, []);

  const interactive = onClick != null;
  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (!onClick) return;
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onClick();
      }
    },
    [onClick],
  );

  return (
    <div
      ref={ref}
      className={`${className ?? ""}${interactive ? " cursor-pointer" : ""}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      onKeyDown={interactive ? handleKeyDown : undefined}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      style={{
        transition: "transform 400ms cubic-bezier(0.2, 0.8, 0.2, 1)",
        willChange: shouldReduceMotion ? undefined : "transform",
      }}
    >
      {children}
    </div>
  );
}

export const AnimatedNumber = React.memo(function AnimatedNumber({
  value,
  duration = 0.6,
  decimals = 0,
  prefix = "",
  suffix = "",
  className,
}: {
  value: number;
  duration?: number;
  /** Number of decimal places to display */
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}) {
  const shouldReduceMotion = useReducedMotion();
  const [displayed, setDisplayed] = React.useState(value);
  const ref = React.useRef<HTMLSpanElement>(null);
  const prevValue = React.useRef(value);
  const animFrame = React.useRef(0);
  const hasBeenVisible = React.useRef(false);

  React.useEffect(() => {
    if (shouldReduceMotion) {
      cancelAnimationFrame(animFrame.current);
      setDisplayed(value);
      prevValue.current = value;
      return;
    }
    const el = ref.current;
    if (!el) return;

    const runAnimation = () => {
      cancelAnimationFrame(animFrame.current);
      const from = prevValue.current;
      const to = value;
      if (from === to) {
        setDisplayed(to);
        prevValue.current = to;
        return;
      }
      const start = performance.now();
      const durationMs = duration * 1000;
      const animate = (now: number) => {
        const progress = Math.min((now - start) / durationMs, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = from + (to - from) * eased;
        setDisplayed(decimals > 0 ? parseFloat(current.toFixed(decimals)) : Math.round(current));
        if (progress < 1) {
          animFrame.current = requestAnimationFrame(animate);
        } else {
          prevValue.current = to;
        }
      };
      animFrame.current = requestAnimationFrame(animate);
    };

    if (hasBeenVisible.current) {
      runAnimation();
      return () => cancelAnimationFrame(animFrame.current);
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          hasBeenVisible.current = true;
          runAnimation();
          observer.disconnect();
        }
      },
      { threshold: 0.3 },
    );
    observer.observe(el);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(animFrame.current);
    };
  }, [value, duration, decimals, shouldReduceMotion]);

  const formatted =
    decimals > 0 ? displayed.toFixed(decimals) : displayed.toLocaleString(getActiveIntlLocale());
  return (
    <span ref={ref} className={className}>
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
});
