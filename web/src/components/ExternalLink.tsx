import type { ReactNode } from "react";

import { ExternalIcon } from "./Icons";

interface Props {
  href: string;
  children: ReactNode;
  className?: string;
  /** Show the small "opens elsewhere" arrow after the text. */
  icon?: boolean;
}

/** A link that opens in a new tab and says so to assistive technology. */
export function ExternalLink({ href, children, className, icon = false }: Props) {
  return (
    <a href={href} className={className} target="_blank" rel="noreferrer noopener">
      {children}
      {icon ? <ExternalIcon size={13} /> : null}
      <span className="visually-hidden"> (opens in a new tab)</span>
    </a>
  );
}
