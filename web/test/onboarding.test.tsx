import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/App";
import { PUBLIC_DEMO_TEXT } from "../src/components/AccessNotice";
import { DOCS_URL, GITHUB_URL } from "../src/links";
import { STORAGE_KEY } from "../src/state/settings";
import {
  EXAMPLE,
  HEALTH,
  doneEvent,
  install,
  json,
  resetViewport,
  setViewport,
  sse,
} from "./helpers";

const KEY_CARD = "Add an API key to start asking";

function answerStream(requestId: string) {
  return () =>
    sse(['event: token\ndata: {"delta": "Pass status_code=201."}\n\n', doneEvent(requestId)], {
      "x-request-id": requestId,
    });
}

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
  resetViewport();
});

describe("first-run screen", () => {
  it("says what the assistant is, shows service status and links", async () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ apiKey: "fa_test" }));
    install({ "/health": () => json(HEALTH) });
    render(<App />);

    const welcome = screen.getByRole("region", { name: "Ask anything about FastAPI" });
    expect(within(welcome).getByText(/small language model specialised in FastAPI/)).toBeInTheDocument();
    expect(await within(welcome).findByText("Model v2026.11.12")).toBeInTheDocument();
    expect(within(welcome).getByText("Fine-tuned")).toBeInTheDocument();
    expect(within(welcome).getByText("Ready")).toBeInTheDocument();

    expect(within(welcome).getByRole("link", { name: /^Docs/ })).toHaveAttribute("href", DOCS_URL);
    expect(within(welcome).getByRole("link", { name: /^GitHub/ })).toHaveAttribute("href", GITHUB_URL);
    expect(within(welcome).getByRole("link", { name: /^API docs/ })).toHaveAttribute(
      "href",
      `${window.location.origin}/docs`,
    );
    expect(within(welcome).getAllByRole("button")).toHaveLength(6);
    // Neither access notice: a key is saved and the server did not say it is public.
    expect(screen.queryByText(PUBLIC_DEMO_TEXT)).toBeNull();
    expect(screen.queryByRole("heading", { name: KEY_CARD })).toBeNull();
  });

  it("reports the base model while weights are loading", async () => {
    install({ "/health": () => json({ ...HEALTH, uses_adapter: false, model_loaded: false }) });
    render(<App />);

    const welcome = screen.getByRole("region", { name: "Ask anything about FastAPI" });
    expect(await within(welcome).findByText("Base model")).toBeInTheDocument();
    expect(within(welcome).getByText("Loading")).toBeInTheDocument();
  });
});

describe("access onboarding", () => {
  it("public: asks without a key and says so", async () => {
    const calls = install({
      "/health": () => json({ ...HEALTH, access: "public" }),
      "/generate": answerStream("pub-1"),
    });
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText(PUBLIC_DEMO_TEXT)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: KEY_CARD })).toBeNull();
    expect(screen.getByText(/Optional on this server/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: EXAMPLE }));

    const answer = await screen.findByLabelText("Assistant answer");
    await waitFor(() => expect(within(answer).getByText(/status_code=201/)).toBeInTheDocument());
    const generate = calls.find((c) => c.url === "/generate");
    expect(new Headers(generate?.init.headers).has("x-api-key")).toBe(false);
    // The live region reports completion once, not every token.
    expect(await screen.findByText("Answer complete, with 1 source.")).toBeInTheDocument();
  });

  it("key: explains how to add a key instead of sending a request that would fail", async () => {
    const calls = install({
      "/health": () => json({ ...HEALTH, access: "key" }),
      "/generate": answerStream("key-1"),
    });
    const user = userEvent.setup();
    render(<App />);

    const card = await screen.findByRole("region", { name: KEY_CARD });
    expect(within(card).getByText(/Open Settings and paste your key/)).toBeInTheDocument();
    expect(screen.getByText("Add an API key in Settings to ask a question.")).toBeInTheDocument();
    expect(screen.queryByText(PUBLIC_DEMO_TEXT)).toBeNull();

    // Picking an example keeps the question and moves to the key field; nothing is sent.
    await user.click(screen.getByRole("button", { name: EXAMPLE }));
    const keyInput = screen.getByLabelText(/^API key/);
    await waitFor(() => expect(keyInput).toHaveFocus());
    expect(screen.getByLabelText("Your FastAPI question")).toHaveValue(EXAMPLE);
    expect(calls.some((c) => c.url === "/generate")).toBe(false);

    // Once a key is entered the card goes away and the kept question can be sent.
    await user.type(keyInput, "fa_new");
    expect(screen.queryByRole("region", { name: KEY_CARD })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(calls.some((c) => c.url === "/generate")).toBe(true));
    const generate = calls.find((c) => c.url === "/generate");
    expect(new Headers(generate?.init.headers).get("x-api-key")).toBe("fa_new");
  });

  it("key card button focuses the API key field", async () => {
    install({ "/health": () => json({ ...HEALTH, access: "key" }) });
    const user = userEvent.setup();
    render(<App />);

    const card = await screen.findByRole("region", { name: KEY_CARD });
    await user.click(within(card).getByRole("button", { name: "Add API key" }));
    expect(screen.getByLabelText(/^API key/)).toHaveFocus();
  });

  it("absent access (older server): treated as key-required", async () => {
    const calls = install({ "/health": () => json(HEALTH), "/generate": answerStream("old-1") });
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole("region", { name: KEY_CARD })).toBeInTheDocument();
    expect(screen.queryByText(PUBLIC_DEMO_TEXT)).toBeNull();
    await user.type(screen.getByLabelText("Your FastAPI question"), "What is Depends?{Enter}");
    expect(calls.some((c) => c.url === "/generate")).toBe(false);
    expect(screen.getByLabelText("Your FastAPI question")).toHaveValue("What is Depends?");
  });

  it("key with a saved key: no card, requests carry the key", async () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ apiKey: "fa_saved" }));
    const calls = install({
      "/health": () => json({ ...HEALTH, access: "key" }),
      "/generate": answerStream("saved-1"),
    });
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("Ready")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: KEY_CARD })).toBeNull();
    await user.click(screen.getByRole("button", { name: EXAMPLE }));
    await waitFor(() => expect(calls.some((c) => c.url === "/generate")).toBe(true));
    const generate = calls.find((c) => c.url === "/generate");
    expect(new Headers(generate?.init.headers).get("x-api-key")).toBe("fa_saved");
  });

  it("key on a narrow screen: opens the settings drawer at the key field", async () => {
    setViewport("narrow");
    install({ "/health": () => json({ ...HEALTH, access: "key" }) });
    const user = userEvent.setup();
    render(<App />);

    await screen.findByRole("region", { name: KEY_CARD });
    await user.click(screen.getByRole("button", { name: EXAMPLE }));

    const dialog = await screen.findByRole("dialog", { name: "Settings and tools" });
    await waitFor(() => expect(within(dialog).getByLabelText(/^API key/)).toHaveFocus());
  });
});
