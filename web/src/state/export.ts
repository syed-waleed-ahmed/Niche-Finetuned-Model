/**
 * Export the conversation as a Markdown file (#42). Pure functions, plus one that
 * triggers the browser download, so the formatting is easy to test.
 */

import { finishedPairs, type Turn } from "./conversation";

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

/** `fastapi-assistant-YYYY-MM-DD.md`, in the viewer's local date. */
export function exportFileName(now: Date): string {
  return `fastapi-assistant-${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}.md`;
}

/**
 * Each finished question and answer, oldest first. Answers are already Markdown and
 * are kept as they are; an answer that is still being written is left out.
 */
export function conversationToMarkdown(turns: Turn[], now: Date): string {
  const parts = [
    "# FastAPI Assistant conversation",
    "",
    `Exported ${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}. Answers come from a small model; check them before use.`,
  ];
  const pairs = finishedPairs(turns);
  for (let i = 0; i + 1 < pairs.length; i += 2) {
    const question = pairs[i];
    const answer = pairs[i + 1];
    if (!question || answer?.role !== "assistant") continue;
    parts.push("", "---", "", `**You:** ${question.content.trim()}`, "");
    if (answer.content.trim()) parts.push(answer.content.trim());
    if (answer.status === "error") parts.push(`_Error: ${answer.error ?? "the request failed"}_`);
    else if (answer.status === "stopped") parts.push("_Stopped before the answer was complete._");
    if (answer.modelVersion) parts.push("", `_Model: ${answer.modelVersion}_`);
  }
  return parts.join("\n") + "\n";
}

/** True when there is at least one finished answer to export. */
export function canExport(turns: Turn[]): boolean {
  return finishedPairs(turns).length > 0;
}

/** Save the conversation through a temporary `<a download>` link. */
export function downloadConversation(turns: Turn[], now: Date = new Date()): void {
  const blob = new Blob([conversationToMarkdown(turns, now)], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = exportFileName(now);
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
