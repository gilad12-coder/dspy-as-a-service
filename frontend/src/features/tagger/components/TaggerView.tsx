"use client";

import type { TaggerSessionDetail } from "@/shared/lib/api";
import { useTagger } from "../hooks/use-tagger";
import { TaggerSetup } from "./TaggerSetup";
import { TaggerAnnotation } from "./TaggerAnnotation";

export function TaggerView({ initialSession }: { initialSession?: TaggerSessionDetail | null }) {
  const tagger = useTagger(initialSession);

  if (tagger.phase === "setup") {
    return <TaggerSetup onStart={tagger.startAnnotating} />;
  }

  if (!tagger.config) return null;

  return (
    <TaggerAnnotation
      config={tagger.config}
      data={tagger.data}
      columns={tagger.columns}
      annotations={tagger.annotations}
      currentIndex={tagger.currentIndex}
      taggedCount={tagger.taggedCount}
      onNavigate={tagger.navigate}
      onGoTo={tagger.goTo}
      onJumpUntagged={tagger.jumpToUntagged}
      onToggleBinary={tagger.toggleBinary}
      onToggleCategory={tagger.toggleCategory}
      onSetFreetext={tagger.setFreetext}
      onBack={tagger.backToSetup}
    />
  );
}
