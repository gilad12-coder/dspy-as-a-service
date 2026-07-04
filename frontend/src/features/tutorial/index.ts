export { TutorialOverlay } from "./components/tutorial-overlay";
export { TutorialMenu } from "./components/tutorial-menu";
export { TutorialProvider, useTutorialContext } from "./components/tutorial-provider";
export { ConceptsGuide } from "./components/concepts-guide.lazy";
export {
  consumePendingCompareDemo,
  consumePendingCompareExamples,
  registerTutorialHook,
  registerTutorialQuery,
} from "./lib/bridge";
// Demo fixtures are deliberately NOT re-exported here: this barrel sits in
// the always-mounted app shell's import graph, so re-exporting
// ./lib/demo-data (~1.5k lines of demo payloads) would pull it into the
// shared first-load chunk of every route. Consumers deep-import
// @/features/tutorial/lib/demo-data so the fixtures land in their own
// route's chunk instead.
