"use client";

import * as React from "react";
import { useReducer, useEffect, useCallback, useRef, createContext, useContext } from "react";
import type { TutorialTrack, TutorialStep } from "../lib/steps";
import {
  getLoadedTrack,
  loadStepsModule,
  resetLoadedTutorialOneShotState,
} from "../lib/steps-loader";

interface TutorialState {
  activeTrack: TutorialTrack | null;
  currentStepIndex: number;
  isVisible: boolean;
  isMenuOpen: boolean;
  isAutoPlaying: boolean;
  completedTracks: Set<TutorialTrack>;
  /** Direction of the last navigation. Used by the auto-skip path so a
   * missing target while walking backward calls prevStep(), not nextStep()
   * — otherwise Backspace can race past the last step into completeTrack(). */
  lastDirection: "forward" | "backward";
}

type TutorialAction =
  | { type: "START_TRACK"; track: TutorialTrack }
  | { type: "NEXT_STEP" }
  | { type: "PREV_STEP" }
  | { type: "GO_TO_STEP"; index: number }
  | { type: "EXIT_TUTORIAL" }
  | { type: "COMPLETE_TRACK" }
  | { type: "OPEN_MENU" }
  | { type: "CLOSE_MENU" }
  | { type: "RESET_ALL" }
  | { type: "TOGGLE_AUTO_PLAY" }
  | { type: "SET_AUTO_PLAY"; value: boolean }
  | { type: "LOAD_STATE"; state: Partial<TutorialState> };

function tutorialReducer(state: TutorialState, action: TutorialAction): TutorialState {
  switch (action.type) {
    case "START_TRACK":
      return {
        ...state,
        activeTrack: action.track,
        currentStepIndex: 0,
        isVisible: true,
        isMenuOpen: false,
        lastDirection: "forward",
      };
    case "NEXT_STEP": {
      if (!state.activeTrack) return state;
      const track = getLoadedTrack(state.activeTrack);
      if (!track) return state;
      const nextIndex = state.currentStepIndex + 1;
      if (nextIndex >= track.steps.length) {
        return {
          ...state,
          isVisible: false,
          completedTracks: new Set([...state.completedTracks, state.activeTrack]),
          lastDirection: "forward",
        };
      }
      return { ...state, currentStepIndex: nextIndex, lastDirection: "forward" };
    }
    case "PREV_STEP":
      return state.activeTrack
        ? {
            ...state,
            currentStepIndex: Math.max(0, state.currentStepIndex - 1),
            lastDirection: "backward",
          }
        : state;
    case "GO_TO_STEP": {
      if (!state.activeTrack) return state;
      const t = getLoadedTrack(state.activeTrack);
      if (!t) return state;
      return {
        ...state,
        currentStepIndex: Math.max(0, Math.min(action.index, t.steps.length - 1)),
      };
    }
    case "EXIT_TUTORIAL":
      return { ...state, isVisible: false, isMenuOpen: false };
    case "COMPLETE_TRACK":
      return state.activeTrack
        ? {
            ...state,
            isVisible: false,
            completedTracks: new Set([...state.completedTracks, state.activeTrack]),
          }
        : state;
    case "OPEN_MENU":
      return { ...state, isMenuOpen: true, isVisible: false };
    case "CLOSE_MENU":
      return { ...state, isMenuOpen: false };
    case "TOGGLE_AUTO_PLAY":
      return { ...state, isAutoPlaying: !state.isAutoPlaying };
    case "SET_AUTO_PLAY":
      return { ...state, isAutoPlaying: action.value };
    case "RESET_ALL":
      return {
        activeTrack: null,
        currentStepIndex: 0,
        isVisible: false,
        isMenuOpen: false,
        isAutoPlaying: false,
        completedTracks: new Set(),
        lastDirection: "forward",
      };
    case "LOAD_STATE":
      return {
        ...state,
        ...action.state,
        completedTracks: new Set(action.state.completedTracks || []),
      };
    default:
      return state;
  }
}

const initialState: TutorialState = {
  activeTrack: null,
  currentStepIndex: 0,
  isVisible: false,
  isMenuOpen: false,
  isAutoPlaying: false,
  completedTracks: new Set(),
  lastDirection: "forward",
};

const STORAGE_KEY = "skynet-tutorial-state";

interface TutorialContextValue {
  state: TutorialState;
  currentStep: TutorialStep | undefined;
  startTrack: (track: TutorialTrack) => void;
  nextStep: () => void;
  prevStep: () => void;
  goToStep: (index: number) => void;
  exitTutorial: () => void;
  completeTrack: () => void;
  startDeepDive: () => void;
  closeMenu: () => void;
  resetAll: () => void;
  toggleAutoPlay: () => void;
}

const TutorialContext = createContext<TutorialContextValue | null>(null);

export function TutorialProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(tutorialReducer, initialState);

  // Restore completedTracks on mount. We deliberately do NOT restore
  // activeTrack/currentStepIndex/isVisible — a fresh page load should
  // start with the tour closed, even if it was open at unload.
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        const completed = new Set<TutorialTrack>(parsed.completedTracks || []);
        dispatch({
          type: "LOAD_STATE",
          state: { completedTracks: completed },
        });
      }
    } catch {
      // Intentionally swallow: localStorage can throw in private-mode
      // Safari or when disabled by the user, and JSON.parse can throw on
      // corrupted state. Falling back to defaults is the right behavior;
      // a stale completion flag isn't worth surfacing to the user.
    }

    // Auto-start tutorial in auto-play mode via URL param
    const params = new URLSearchParams(window.location.search);
    if (params.get("tutorial") === "autoplay") {
      setTimeout(() => {
        void loadStepsModule().then(() => {
          dispatch({ type: "SET_AUTO_PLAY", value: true });
          dispatch({ type: "START_TRACK", track: "deep-dive" as TutorialTrack });
        });
      }, 1000);
    }
  }, []);

  // Persist only completed tracks. Ephemeral session state (active track,
  // step index, visibility) intentionally lives in memory only.
  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          completedTracks: Array.from(state.completedTracks),
        }),
      );
    } catch {
      // Intentionally swallow: setItem throws on quota exceeded or when
      // storage is disabled. Tutorial completion persistence is best-effort.
    }
  }, [state.completedTracks]);

  const startTrack = useCallback((track: TutorialTrack) => {
    // The steps module loads on demand (it is deliberately absent from the
    // shared first-load chunk); START_TRACK only dispatches once the load
    // resolves, so every synchronous getLoadedTrack() read that follows
    // hits a loaded module.
    void loadStepsModule().then((steps) => {
      // Clear per-tour ephemeral flags (e.g. dd-detail-header splash
      // one-shot) so a fresh tour run gets a fresh splash, not the leftover
      // state from the previous run.
      steps.resetTutorialOneShotState();
      dispatch({ type: "START_TRACK", track });
    });
  }, []);
  const nextStep = useCallback(() => dispatch({ type: "NEXT_STEP" }), []);
  const prevStep = useCallback(() => dispatch({ type: "PREV_STEP" }), []);
  const goToStep = useCallback((index: number) => dispatch({ type: "GO_TO_STEP", index }), []);
  const exitTutorial = useCallback(() => dispatch({ type: "EXIT_TUTORIAL" }), []);
  const completeTrack = useCallback(() => dispatch({ type: "COMPLETE_TRACK" }), []);
  // We have a single track, so the help button starts it directly rather
  // than opening a one-option chooser. The first step's beforeShow
  // (ensureDashboard) handles client-side navigation to "/" via the
  // routerPush bridge hook.
  const startDeepDive = useCallback(() => {
    startTrack("deep-dive" as TutorialTrack);
  }, [startTrack]);
  const closeMenu = useCallback(() => dispatch({ type: "CLOSE_MENU" }), []);
  const resetAll = useCallback(() => {
    resetLoadedTutorialOneShotState();
    dispatch({ type: "RESET_ALL" });
    localStorage.removeItem(STORAGE_KEY);
  }, []);
  const toggleAutoPlay = useCallback(() => dispatch({ type: "TOGGLE_AUTO_PLAY" }), []);

  // Notify listeners (e.g. dashboard demo overlay) when tutorial closes
  const prevVisible = useRef(state.isVisible);
  useEffect(() => {
    if (prevVisible.current && !state.isVisible) {
      window.dispatchEvent(new Event("tutorial-exited"));
    }
    prevVisible.current = state.isVisible;
  }, [state.isVisible]);

  const currentStep = state.activeTrack
    ? getLoadedTrack(state.activeTrack)?.steps[state.currentStepIndex]
    : undefined;

  return (
    <TutorialContext.Provider
      value={{
        state,
        currentStep,
        startTrack,
        nextStep,
        prevStep,
        goToStep,
        exitTutorial,
        completeTrack,
        startDeepDive,
        closeMenu,
        resetAll,
        toggleAutoPlay,
      }}
    >
      {children}
    </TutorialContext.Provider>
  );
}

export function useTutorialContext() {
  const context = useContext(TutorialContext);
  if (!context) {
    throw new Error("useTutorialContext must be used within TutorialProvider");
  }
  return context;
}
