import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { KeyRequiredCard } from "./components/AccessNotice";
import { Composer } from "./components/Composer";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { RetryBanner } from "./components/RetryBanner";
import { SIDE_PANEL_ID, SidePanel, type Tab } from "./components/SidePanel";
import { Welcome } from "./components/Welcome";
import { useConversation } from "./hooks/useConversation";
import { useHealth } from "./hooks/useHealth";
import { NARROW_QUERY, useMediaQuery } from "./hooks/useMediaQuery";
import { useSettings } from "./hooks/useSettings";
import { useTheme } from "./hooks/useTheme";
import { accessMode } from "./state/access";
import type { Turn } from "./state/conversation";

type FocusRequest = "api-key" | "close" | null;

/** What the polite live region says about the latest answer: transitions, not tokens. */
export function announcementFor(turn: Turn | undefined): string {
  if (!turn || turn.role !== "assistant") return "";
  switch (turn.status) {
    case "pending":
      return "Waiting for the model.";
    case "streaming":
      return "Generating an answer.";
    case "done": {
      const n = turn.sources.length;
      return n ? `Answer complete, with ${n} source${n === 1 ? "" : "s"}.` : "Answer complete.";
    }
    case "stopped":
      return "Answer stopped.";
    case "error":
      return ""; // The error itself is a role="alert" inside the answer.
  }
}

export function App() {
  const [settings, updateSettings] = useSettings();
  const [theme, setTheme] = useTheme();
  const [draft, setDraft] = useState("");
  const [tab, setTab] = useState<Tab>("settings");
  const narrow = useMediaQuery(NARROW_QUERY);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [focusRequest, setFocusRequest] = useState<FocusRequest>(null);
  const settingsButtonRef = useRef<HTMLButtonElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const wasOpenRef = useRef(false);

  const clientConfig = useMemo(
    () => ({ baseUrl: settings.baseUrl, apiKey: settings.apiKey }),
    [settings.baseUrl, settings.apiKey],
  );
  const health = useHealth(clientConfig);
  const conversation = useConversation(settings);

  const access = accessMode(health.health);
  // Ask for a key before the first request instead of letting it fail with a 401.
  const needsKey = access === "key" && !settings.apiKey;
  const modal = narrow && drawerOpen;

  /** Show the settings: the drawer on narrow screens, the side panel otherwise. */
  const openSettings = (focusKey: boolean) => {
    setTab("settings");
    if (narrow) {
      if (!drawerOpen) {
        const active = document.activeElement;
        returnFocusRef.current =
          active instanceof HTMLElement && active !== document.body ? active : null;
        setDrawerOpen(true);
      }
      setFocusRequest(focusKey ? "api-key" : "close");
      return;
    }
    if (focusKey) setFocusRequest("api-key");
  };

  const closeDrawer = useCallback(() => setDrawerOpen(false), []);

  // A 401 is almost always a missing or wrong key: bring the settings into view.
  useEffect(() => {
    if (conversation.authError) openSettings(true);
  }, [conversation.authError]);

  // Focus moves after the render that shows its target (tab switch, drawer opening).
  useEffect(() => {
    if (!focusRequest) return;
    const target =
      focusRequest === "api-key" ? document.getElementById("api-key") : closeButtonRef.current;
    target?.focus();
    setFocusRequest(null);
  }, [focusRequest]);

  // Closing the drawer returns focus to whatever opened it (normally the header button).
  useEffect(() => {
    if (drawerOpen) {
      wasOpenRef.current = true;
      return;
    }
    if (!wasOpenRef.current) return;
    wasOpenRef.current = false;
    const target = returnFocusRef.current;
    returnFocusRef.current = null;
    (target?.isConnected ? target : settingsButtonRef.current)?.focus();
  }, [drawerOpen]);

  useEffect(() => {
    if (!modal) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setDrawerOpen(false);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [modal]);

  // Widening the window turns the drawer back into the inline panel.
  useEffect(() => {
    if (!narrow) setDrawerOpen(false);
  }, [narrow]);

  const displayBaseUrl =
    settings.baseUrl || (typeof window !== "undefined" ? window.location.origin : "");
  const apiDocsUrl = `${displayBaseUrl}/docs`;

  const send = (question: string) => {
    if (needsKey) {
      // Keep the question and point at the missing key; nothing is sent.
      setDraft(question);
      openSettings(true);
      return;
    }
    setDraft("");
    void conversation.send(question);
  };

  const lastTurn = conversation.turns[conversation.turns.length - 1];
  const announcement = announcementFor(lastTurn);
  const empty = conversation.turns.length === 0;

  return (
    <div className="app">
      <Header
        health={health}
        apiDocsUrl={apiDocsUrl}
        theme={theme}
        onTheme={setTheme}
        onOpenSettings={() => openSettings(!narrow)}
        drawer={narrow ? { open: drawerOpen, controls: SIDE_PANEL_ID } : null}
        settingsRef={settingsButtonRef}
        inert={modal}
      />
      <div className="layout">
        <main className="conversation" aria-label="Conversation" inert={modal}>
          <MessageList
            turns={conversation.turns}
            onRate={conversation.rate}
            welcome={
              <Welcome
                health={health}
                access={access}
                needsKey={needsKey}
                apiDocsUrl={apiDocsUrl}
                onPick={send}
                onAddKey={() => openSettings(true)}
              />
            }
          />
          <div className="dock">
            {conversation.retry || (needsKey && !empty) ? (
              <div className="dock__notices">
                {conversation.retry ? (
                  <RetryBanner retry={conversation.retry} onCancel={conversation.stop} />
                ) : null}
                {needsKey && !empty ? (
                  <KeyRequiredCard compact onAddKey={() => openSettings(true)} />
                ) : null}
              </div>
            ) : null}
            <Composer
              draft={draft}
              onDraft={setDraft}
              onSend={send}
              onStop={conversation.stop}
              busy={conversation.busy}
              disabled={false}
              hint={needsKey ? "Add an API key in Settings to ask a question." : null}
            />
          </div>
        </main>
        {narrow ? (
          <div
            className={`backdrop ${drawerOpen ? "backdrop--open" : ""}`}
            onClick={closeDrawer}
            aria-hidden="true"
            data-testid="drawer-backdrop"
          />
        ) : null}
        <SidePanel
          tab={tab}
          onTab={setTab}
          settings={settings}
          onSettings={updateSettings}
          health={health}
          access={access}
          lastMeta={conversation.lastMeta}
          draft={draft}
          turnCount={conversation.turns.length}
          onClear={conversation.clear}
          authError={conversation.authError}
          onDismissAuthError={conversation.dismissAuthError}
          displayBaseUrl={displayBaseUrl}
          apiDocsUrl={apiDocsUrl}
          theme={theme}
          onTheme={setTheme}
          variant={narrow ? "drawer" : "inline"}
          open={drawerOpen}
          onClose={closeDrawer}
          closeRef={closeButtonRef}
        />
      </div>
      <div className="visually-hidden" aria-live="polite" aria-atomic="true">
        {announcement}
      </div>
    </div>
  );
}
