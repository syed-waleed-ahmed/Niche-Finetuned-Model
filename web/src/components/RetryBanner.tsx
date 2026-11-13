import { useEffect, useState } from "react";

import type { RetryState } from "../hooks/useConversation";

interface Props {
  retry: RetryState;
  onCancel: () => void;
}

export function RetryBanner({ retry, onCancel }: Props) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(timer);
  }, [retry.untilMs]);

  const remaining = Math.max(0, Math.ceil((retry.untilMs - now) / 1000));
  const reason =
    retry.status === 429
      ? "The service is at capacity (429 with Retry-After)."
      : "The service is not ready (503 with Retry-After).";

  return (
    <div className="banner banner--warn" role="status">
      <span className="banner__text">
        {reason} Retrying in <span className="banner__count">{remaining}s</span> (attempt{" "}
        {retry.attempt + 1} of {retry.maxAttempts}).
      </span>
      <button type="button" className="btn btn--sm btn--quiet" onClick={onCancel}>
        Cancel
      </button>
    </div>
  );
}
