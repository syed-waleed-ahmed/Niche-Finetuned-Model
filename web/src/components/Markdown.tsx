import bash from "highlight.js/lib/languages/bash";
import dockerfile from "highlight.js/lib/languages/dockerfile";
import http from "highlight.js/lib/languages/http";
import ini from "highlight.js/lib/languages/ini";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import plaintext from "highlight.js/lib/languages/plaintext";
import python from "highlight.js/lib/languages/python";
import shell from "highlight.js/lib/languages/shell";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import yaml from "highlight.js/lib/languages/yaml";
import { isValidElement, memo, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import { CopyButton } from "./CopyButton";

// Only the languages a FastAPI assistant produces; the full highlight.js bundle is
// large and the model never emits most of it.
const languages = {
  python,
  py: python,
  bash,
  sh: bash,
  shell,
  console: shell,
  json,
  yaml,
  yml: yaml,
  javascript,
  js: javascript,
  typescript,
  ts: typescript,
  http,
  ini,
  toml: ini,
  dockerfile,
  sql,
  plaintext,
  text: plaintext,
};

const remarkPlugins = [remarkGfm];
const rehypePlugins = [[rehypeHighlight, { languages, detect: false, ignoreMissing: true }] as const];

/** Collect the text of a React subtree (used to copy a highlighted code block). */
export function textOf(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return textOf(node.props.children);
  return "";
}

function languageOf(children: ReactNode): string | null {
  if (!isValidElement<{ className?: string }>(children)) return null;
  const match = /language-([\w-]+)/.exec(children.props.className ?? "");
  return match?.[1] ?? null;
}

const components: Components = {
  pre({ children }) {
    const code = textOf(children).replace(/\n$/, "");
    const lang = languageOf(children);
    return (
      <div className="codeblock">
        <div className="codeblock__bar">
          <span className="codeblock__lang">{lang ?? "code"}</span>
          <CopyButton text={code} ariaLabel="Copy code" className="codeblock__copy" />
        </div>
        <pre>{children}</pre>
      </div>
    );
  },
  a({ children, href }) {
    return (
      <a href={href} target="_blank" rel="noreferrer noopener">
        {children}
      </a>
    );
  },
};

export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins as never}
        components={components}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});
