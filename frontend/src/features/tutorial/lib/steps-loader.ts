/**
 * Lazy gateway to the tutorial step definitions.
 *
 * `steps.ts` (and the demo-data module it pulls in) is ~2.6k lines that the
 * always-mounted TutorialProvider/TutorialOverlay previously dragged into the
 * shared first-load chunk of every route. The provider loads the module here
 * on the first tour start instead; the synchronous getters let the reducer
 * and render paths (which cannot await) read the already-loaded module.
 */
import type { TutorialTrack, TutorialTrackDefinition } from "./steps";

// Structural view of the lazily-imported ./steps module (the inline
// `typeof import(...)` annotation is banned by consistent-type-imports).
interface StepsModule {
  getTrack: (trackId: TutorialTrack) => TutorialTrackDefinition | undefined;
  resetTutorialOneShotState: () => void;
}

let stepsModule: StepsModule | null = null;

export async function loadStepsModule(): Promise<StepsModule> {
  if (!stepsModule) {
    stepsModule = await import("./steps");
  }
  return stepsModule;
}

/**
 * Synchronous read of an already-loaded track. Returns undefined until
 * loadStepsModule() has resolved — callers only reach this after a tour
 * start, which awaits the load first.
 */
export function getLoadedTrack(track: TutorialTrack): TutorialTrackDefinition | undefined {
  return stepsModule?.getTrack(track);
}

/** Clear per-tour one-shot flags, when the steps module was ever loaded. */
export function resetLoadedTutorialOneShotState(): void {
  stepsModule?.resetTutorialOneShotState();
}
