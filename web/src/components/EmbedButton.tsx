import { useState } from "react";
import { embedSnippet } from "../lib/embed";
import { currentUrlState } from "../lib/urlState";
import { useAppStore } from "../state/store";

/** Copies an iframe that shows exactly this state inside another page. */
export function EmbedButton() {
  const [done, setDone] = useState<"idle" | "copied" | "failed">("idle");
  return (
    <button
      type="button"
      className="link-button"
      onClick={() =>
        void navigator.clipboard
          .writeText(embedSnippet(location.origin, currentUrlState(useAppStore.getState())))
          .then(
            () => setDone("copied"),
            () => setDone("failed"),
          )
      }
    >
      {done === "copied" ? "Embed code copied" : done === "failed" ? "Could not copy" : "Embed"}
    </button>
  );
}
