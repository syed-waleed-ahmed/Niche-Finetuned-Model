import { describe, expect, it, vi } from "vitest";

import { newAssistantTurn, type Turn } from "../src/state/conversation";
import {
  canExport,
  conversationToMarkdown,
  downloadConversation,
  exportFileName,
} from "../src/state/export";

const NOW = new Date(2026, 10, 14, 9, 5);

function pair(question: string, answer: string, extra: Partial<ReturnType<typeof newAssistantTurn>> = {}): Turn[] {
  return [
    { id: `u-${question}`, role: "user", content: question },
    { ...newAssistantTurn("generate", true, 1000), content: answer, status: "done", ...extra },
  ];
}

describe("conversation export (#42)", () => {
  it("names the file after the local date", () => {
    expect(exportFileName(NOW)).toBe("fastapi-assistant-2026-11-14.md");
  });

  it("writes each question and its answer, keeping code blocks as they are", () => {
    const answer = "Set the status code:\n\n```python\n@app.post(\"/items\", status_code=201)\ndef create(): ...\n```";
    const md = conversationToMarkdown(pair("How do I return 201?", answer, { modelVersion: "v1" }), NOW);
    expect(md.startsWith("# FastAPI Assistant conversation\n")).toBe(true);
    expect(md).toContain("Exported 2026-11-14 09:05.");
    expect(md).toContain("**You:** How do I return 201?\n\n" + answer + "\n\n_Model: v1_");
  });

  it("leaves out an answer that is still streaming, and marks stopped and failed ones", () => {
    const streaming: Turn[] = [
      { id: "u-s", role: "user", content: "unfinished" },
      { ...newAssistantTurn("generate", true, 1000), content: "half", status: "streaming" },
    ];
    const md = conversationToMarkdown(
      [
        ...pair("stopped one", "partial", { status: "stopped" }),
        ...pair("failed one", "", { status: "error", error: "Rate limit exceeded" }),
        ...streaming,
      ],
      NOW,
    );
    expect(md).toContain("partial\n_Stopped before the answer was complete._");
    expect(md).toContain("**You:** failed one\n\n_Error: Rate limit exceeded_");
    expect(md).not.toContain("unfinished");
    expect(md).not.toContain("half");
  });

  it("has nothing to export until an answer is finished", () => {
    expect(canExport([])).toBe(false);
    expect(canExport(pair("q", "a"))).toBe(true);
  });

  it("downloads through a temporary link and releases the URL", () => {
    vi.useFakeTimers();
    const create = vi.fn(() => "blob:test");
    const revoke = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL: create, revokeObjectURL: revoke });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    try {
      downloadConversation(pair("q", "a"), NOW);
      expect(create).toHaveBeenCalledOnce();
      expect(click).toHaveBeenCalledOnce();
      const link = click.mock.contexts[0] as HTMLAnchorElement;
      expect(link.download).toBe("fastapi-assistant-2026-11-14.md");
      expect(document.querySelector("a[download]")).toBeNull();
      vi.runAllTimers();
      expect(revoke).toHaveBeenCalledWith("blob:test");
    } finally {
      click.mockRestore();
      vi.unstubAllGlobals();
      vi.useRealTimers();
    }
  });
});
