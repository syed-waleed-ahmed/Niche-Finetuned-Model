// Stream an answer with the built-in fetch: no dependencies, Node.js 22 or a browser.
//
//   export ASSISTANT_URL=https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev
//   export ASSISTANT_API_KEY=fa_...   # leave unset for a public demo, which needs no key
//   node examples/fetch_stream.mjs "How do I return a 201 from a POST endpoint?"
//
// It calls the OpenAI-compatible POST /v1/chat/completions with `stream: true`. The
// answer arrives as server-sent events: frames separated by a blank line, each with a
// `data:` line holding one JSON chunk, and a final `data: [DONE]`.

const baseUrl = (process.env.ASSISTANT_URL ?? "http://localhost:8000").replace(/\/+$/, "");
// Read the key from the environment; never write one into code.
const apiKey = process.env.ASSISTANT_API_KEY?.trim();
const question = process.argv.slice(2).join(" ") || "What does Depends() do in FastAPI?";

/** A plain sentence for the failures a caller is most likely to meet. */
function explain(response) {
  const retryAfter = response.headers.get("retry-after");
  const wait = retryAfter ? ` Try again in ${retryAfter} s.` : "";
  switch (response.status) {
    case 401:
      return apiKey
        ? "The server rejected the API key. Check ASSISTANT_API_KEY."
        : "This server needs an API key. Set ASSISTANT_API_KEY.";
    case 413:
      return "The request body is too large (the limit is 64 KiB). Send a shorter question.";
    case 429:
      return `Rate limited, or the generation queue is full.${wait}`;
    case 503:
      return `The model is still loading, or the queue wait timed out.${wait}`;
    default:
      return "The request failed.";
  }
}

/** Yield the `data:` payload of each server-sent event in a fetch response body. */
async function* sseData(body) {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const bytes of body) {
    // `stream: true` keeps a multi-byte character that is split across two chunks intact.
    buffer += decoder.decode(bytes, { stream: true }).replaceAll("\r\n", "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) yield data;
    }
  }
}

async function main() {
  const headers = { "content-type": "application/json", accept: "text/event-stream" };
  if (apiKey) headers.authorization = `Bearer ${apiKey}`;

  const response = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: "fastapi-assistant",
      messages: [{ role: "user", content: question }],
      max_tokens: 128,
      stream: true,
    }),
  });
  if (!response.ok) {
    const detail = (await response.text()).slice(0, 300);
    console.error(`HTTP ${response.status}: ${explain(response)}`);
    if (detail) console.error(detail);
    return 1;
  }

  let finish = null;
  for await (const data of sseData(response.body)) {
    if (data === "[DONE]") break;
    const chunk = JSON.parse(data);
    // A failure after the stream has started arrives as a frame, not as a status code.
    if (chunk.error) {
      console.error(`\nerror: ${chunk.error.message}`);
      return 1;
    }
    const choice = chunk.choices?.[0];
    if (choice?.delta?.content) process.stdout.write(choice.delta.content);
    if (choice?.finish_reason) finish = choice.finish_reason;
  }
  console.log(`\n[finish=${finish ?? "unknown"}]`);
  return 0;
}

try {
  process.exitCode = await main();
} catch (err) {
  // fetch rejects with a TypeError when the server cannot be reached at all.
  console.error(`Could not reach ${baseUrl}: ${err.cause?.message ?? err.message}`);
  process.exitCode = 1;
}
