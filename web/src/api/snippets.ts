/**
 * Copyable client snippets generated from the current question and settings.
 * The API key is always a placeholder: the real key never leaves the browser.
 */

export type EndpointMode = "generate" | "chat";

export interface SnippetInput {
  baseUrl: string;
  question: string;
  mode: EndpointMode;
  stream: boolean;
  temperature: number;
  topP: number;
  maxNewTokens: number;
  useRetrieval: boolean | null;
}

const PLACEHOLDER_QUESTION = "How do I return a 201 from a POST endpoint?";

function question(input: SnippetInput): string {
  return input.question.trim() || PLACEHOLDER_QUESTION;
}

export function nativeBody(input: SnippetInput): Record<string, unknown> {
  const body: Record<string, unknown> = {
    question: question(input),
    temperature: input.temperature,
    top_p: input.topP,
    max_new_tokens: input.maxNewTokens,
  };
  if (input.stream) body["stream"] = true;
  if (input.useRetrieval !== null) body["use_retrieval"] = input.useRetrieval;
  return body;
}

export function chatBody(input: SnippetInput): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: "fastapi-assistant",
    messages: [{ role: "user", content: question(input) }],
    temperature: input.temperature,
    top_p: input.topP,
    max_tokens: input.maxNewTokens,
  };
  if (input.stream) body["stream"] = true;
  return body;
}

export function endpointPath(mode: EndpointMode): string {
  return mode === "generate" ? "/generate" : "/v1/chat/completions";
}

function bodyFor(input: SnippetInput): Record<string, unknown> {
  return input.mode === "generate" ? nativeBody(input) : chatBody(input);
}

export function curlSnippet(input: SnippetInput): string {
  const json = JSON.stringify(bodyFor(input), null, 2).split("'").join(String.raw`'\''`);
  const flags = input.stream ? "-sN" : "-s";
  return [
    `curl ${flags} "${input.baseUrl}${endpointPath(input.mode)}" \\`,
    `  -H "X-API-Key: $API_KEY" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '${json}'`,
  ].join("\n");
}

export function pythonRequestsSnippet(input: SnippetInput): string {
  const lines = [
    "import os",
    "import requests",
    "",
    `BASE_URL = ${JSON.stringify(input.baseUrl)}`,
    'headers = {"X-API-Key": os.environ["API_KEY"]}',
    `payload = ${toPython(bodyFor(input))}`,
    "",
  ];
  const url = `f"{BASE_URL}${endpointPath(input.mode)}"`;
  if (input.stream) {
    lines.push(
      `with requests.post(${url}, json=payload, headers=headers, stream=True) as r:`,
      "    r.raise_for_status()",
      "    for line in r.iter_lines():",
      '        if line.startswith(b"data: "):',
      "            print(line[6:].decode())",
    );
  } else {
    lines.push(
      `r = requests.post(${url}, json=payload, headers=headers)`,
      "r.raise_for_status()",
      input.mode === "generate"
        ? 'print(r.json()["answer"])'
        : 'print(r.json()["choices"][0]["message"]["content"])',
    );
  }
  return lines.join("\n");
}

export function pythonOpenAISnippet(input: SnippetInput): string {
  const lines = [
    "import os",
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url=${JSON.stringify(`${input.baseUrl}/v1`)}, api_key=os.environ["API_KEY"])`,
    "",
  ];
  const args = [
    '    model="fastapi-assistant",',
    `    messages=[{"role": "user", "content": ${JSON.stringify(question(input))}}],`,
    `    temperature=${input.temperature},`,
    `    top_p=${input.topP},`,
    `    max_tokens=${input.maxNewTokens},`,
  ];
  if (input.stream) {
    lines.push(
      "for chunk in client.chat.completions.create(",
      ...args,
      "    stream=True,",
      "):",
      '    print(chunk.choices[0].delta.content or "", end="", flush=True)',
    );
  } else {
    lines.push(
      "completion = client.chat.completions.create(",
      ...args,
      ")",
      "print(completion.choices[0].message.content)",
    );
  }
  return lines.join("\n");
}

export function javascriptSnippet(input: SnippetInput): string {
  const body = JSON.stringify(bodyFor(input), null, 2).split("\n").join("\n  ");
  const lines = [
    `const response = await fetch(${JSON.stringify(`${input.baseUrl}${endpointPath(input.mode)}`)}, {`,
    '  method: "POST",',
    '  headers: { "X-API-Key": process.env.API_KEY, "Content-Type": "application/json" },',
    `  body: JSON.stringify(${body}),`,
    "});",
  ];
  if (input.stream) {
    lines.push(
      "const reader = response.body.getReader();",
      "const decoder = new TextDecoder();",
      "for (;;) {",
      "  const { value, done } = await reader.read();",
      "  if (done) break;",
      "  process.stdout.write(decoder.decode(value, { stream: true }));",
      "}",
    );
  } else {
    lines.push(
      input.mode === "generate"
        ? "console.log((await response.json()).answer);"
        : "console.log((await response.json()).choices[0].message.content);",
    );
  }
  return lines.join("\n");
}

/** Render a JSON-like value as a Python literal (True/False/None). */
export function toPython(value: unknown, indent = 0): string {
  const pad = "    ".repeat(indent + 1);
  const close = "    ".repeat(indent);
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((v) => toPython(v, indent)).join(", ")}]`;
  const entries = Object.entries(value as Record<string, unknown>);
  if (entries.length === 0) return "{}";
  const body = entries
    .map(([k, v]) => `${pad}${JSON.stringify(k)}: ${toPython(v, indent + 1)}`)
    .join(",\n");
  return `{\n${body},\n${close}}`;
}
