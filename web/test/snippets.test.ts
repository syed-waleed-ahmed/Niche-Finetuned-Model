import { describe, expect, it } from "vitest";

import {
  curlSnippet,
  javascriptSnippet,
  pythonOpenAISnippet,
  pythonRequestsSnippet,
  toPython,
  type SnippetInput,
} from "../src/api/snippets";

const base: SnippetInput = {
  baseUrl: "https://edge.example",
  question: "What does Depends do?",
  mode: "generate",
  stream: false,
  temperature: 0.2,
  topP: 0.9,
  maxNewTokens: 256,
  useRetrieval: null,
};

describe("snippets", () => {
  it("never embeds a real key, only the placeholder", () => {
    for (const snippet of [curlSnippet, pythonRequestsSnippet, pythonOpenAISnippet, javascriptSnippet]) {
      const text = snippet(base);
      expect(text).toContain("API_KEY");
      expect(text).not.toContain("fa_");
    }
  });

  it("targets the native endpoint with the question and parameters", () => {
    const curl = curlSnippet(base);
    expect(curl).toContain('"https://edge.example/generate"');
    expect(curl).toContain('"question": "What does Depends do?"');
    expect(curl).toContain('"max_new_tokens": 256');
    expect(curl).not.toContain("stream");
    expect(curl).not.toContain("use_retrieval");
  });

  it("switches to the OpenAI-compatible endpoint in chat mode with streaming", () => {
    const curl = curlSnippet({ ...base, mode: "chat", stream: true });
    expect(curl).toContain("curl -sN");
    expect(curl).toContain("/v1/chat/completions");
    expect(curl).toContain('"max_tokens": 256');
    expect(curl).toContain('"stream": true');
  });

  it("escapes single quotes inside the curl body", () => {
    const curl = curlSnippet({ ...base, question: "What's Depends?" });
    expect(curl).toContain(String.raw`What'\''s Depends?`);
  });

  it("falls back to a placeholder question when empty", () => {
    expect(curlSnippet({ ...base, question: "  " })).toContain("How do I return a 201");
  });

  it("renders Python literals and the retrieval flag", () => {
    const py = pythonRequestsSnippet({ ...base, useRetrieval: true, stream: true });
    expect(py).toContain('"use_retrieval": True');
    expect(py).toContain("stream=True");
    expect(py).toContain('f"{BASE_URL}/generate"');
    expect(toPython({ a: null, b: [1, false], c: "x" })).toBe(
      '{\n    "a": None,\n    "b": [1, False],\n    "c": "x",\n}',
    );
  });

  it("points the OpenAI SDK at /v1", () => {
    const py = pythonOpenAISnippet({ ...base, stream: true });
    expect(py).toContain('base_url="https://edge.example/v1"');
    expect(py).toContain("stream=True");
    expect(pythonOpenAISnippet(base)).toContain("completion.choices[0].message.content");
  });

  it("produces a fetch example that reads the right field per mode", () => {
    expect(javascriptSnippet(base)).toContain(".answer");
    expect(javascriptSnippet({ ...base, mode: "chat" })).toContain("choices[0].message.content");
    expect(javascriptSnippet({ ...base, stream: true })).toContain("getReader()");
  });
});
