import { useId } from "react";

import { DOCS_URL } from "../links";
import { ExternalLink } from "./ExternalLink";
import { GlobeIcon, KeyIcon } from "./Icons";

export const PUBLIC_DEMO_TEXT = "Public demo: no key needed, a few questions per minute.";

/** Shown when `/health` reports `access: "public"`. */
export function PublicNotice() {
  return (
    <p className="notice notice--public">
      <GlobeIcon size={16} />
      <span>{PUBLIC_DEMO_TEXT}</span>
    </p>
  );
}

/**
 * Shown when the server requires a key (or does not say, as older servers do) and
 * none is saved, so the first question does not have to fail with a 401 to explain it.
 */
export function KeyRequiredCard({
  onAddKey,
  compact = false,
}: {
  onAddKey: () => void;
  compact?: boolean;
}) {
  const titleId = useId();
  return (
    <section className={`key-card ${compact ? "key-card--compact" : ""}`} aria-labelledby={titleId}>
      <span className="key-card__icon">
        <KeyIcon size={18} />
      </span>
      <div className="key-card__body">
        <h3 className="key-card__title" id={titleId}>
          Add an API key to start asking
        </h3>
        <p className="key-card__text">
          This server answers only requests that carry an API key. Open Settings and paste
          your key under Access; it is kept in this browser and sent only to this API.
          {compact ? null : (
            <>
              {" "}
              No key? The <ExternalLink href={DOCS_URL}>documentation</ExternalLink> shows how to
              run the assistant yourself.
            </>
          )}
        </p>
      </div>
      <button type="button" className="btn btn--primary btn--sm key-card__action" onClick={onAddKey}>
        Add API key
      </button>
    </section>
  );
}
