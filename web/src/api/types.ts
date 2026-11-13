/**
 * Request and response shapes of the service, mirroring
 * `src/fastapi_assistant/schemas.py`. Field names are identical to the wire
 * format so responses can be used without mapping.
 */

export interface GenerateRequest {
  question: string;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
  use_retrieval?: boolean;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface Source {
  url: string;
  heading: string;
  score: number;
}

export interface GenerateResponse {
  question: string;
  answer: string;
  model: string;
  model_version: string;
  uses_adapter: boolean;
  finish_reason: string;
  usage: Usage;
  sources: Source[];
  request_id: string;
}

/** Final SSE event of a streamed `/generate` call (`event: done`). */
export interface GenerateDoneEvent {
  finish_reason: string;
  usage: Usage;
  sources: Source[];
  request_id: string;
  model_version: string;
}

/**
 * Who may ask questions: `"public"` (no API key; the edge rate-limits per IP) or
 * `"key"` (an API key is required).
 */
export type AccessMode = "public" | "key";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  uses_adapter: boolean;
  backend: string;
  model_version: string;
  load_state: string;
  /** Absent on older servers, which always required a key. */
  access?: AccessMode;
}

export interface FeedbackRequest {
  request_id: string;
  rating: 1 | -1;
  comment?: string;
}

export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: {
    index: number;
    message: { role: "assistant"; content: string };
    finish_reason: string;
  }[];
  usage: Usage;
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: {
    index: number;
    delta: { role?: string; content?: string };
    finish_reason: string | null;
  }[];
}
