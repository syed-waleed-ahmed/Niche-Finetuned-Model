import { useEffect, useState } from "react";

import { CheckIcon, CopyIcon } from "./Icons";

interface Props {
  text: string;
  label?: string;
  /** Accessible name when the visible label alone is ambiguous ("Copy code"). */
  ariaLabel?: string;
  className?: string;
}

export function CopyButton({ text, label = "Copy", ariaLabel, className = "" }: Props) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");

  useEffect(() => {
    if (state === "idle") return;
    const timer = setTimeout(() => setState("idle"), 1500);
    return () => clearTimeout(timer);
  }, [state]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setState("copied");
    } catch {
      setState("failed");
    }
  };

  return (
    <>
      <button
        type="button"
        className={`btn btn--sm btn--quiet ${className}`}
        onClick={() => void copy()}
        {...(ariaLabel ? { "aria-label": ariaLabel } : {})}
      >
        {state === "copied" ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
        {state === "copied" ? "Copied" : state === "failed" ? "Failed" : label}
      </button>
      {/* A focused button's changed name is rarely read out, so announce the result. */}
      <span className="visually-hidden" aria-live="polite">
        {state === "copied" ? "Copied to the clipboard" : state === "failed" ? "Copy failed" : ""}
      </span>
    </>
  );
}
