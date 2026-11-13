import { useEffect, useMemo, useRef, useState } from "react";

import { Composer } from "./components/Composer";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { RetryBanner } from "./components/RetryBanner";
import { SidePanel, type Tab } from "./components/SidePanel";
import { useConversation } from "./hooks/useConversation";
import { useHealth } from "./hooks/useHealth";
import { useSettings } from "./hooks/useSettings";

export function App() {
  const [settings, updateSettings] = useSettings();
  const [draft, setDraft] = useState("");
  const [tab, setTab] = useState<Tab>("settings");
  const sideRef = useRef<HTMLDivElement>(null);

  const clientConfig = useMemo(
    () => ({ baseUrl: settings.baseUrl, apiKey: settings.apiKey }),
    [settings.baseUrl, settings.apiKey],
  );
  const health = useHealth(clientConfig);
  const conversation = useConversation(settings);

  // A 401 is almost always a missing key: bring the settings into view.
  useEffect(() => {
    if (conversation.authError) openSettings();
  }, [conversation.authError]);

  const openSettings = () => {
    setTab("settings");
    sideRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    document.getElementById("api-key")?.focus();
  };

  const displayBaseUrl =
    settings.baseUrl || (typeof window !== "undefined" ? window.location.origin : "");

  const send = (question: string) => {
    setDraft("");
    void conversation.send(question);
  };

  return (
    <div className="app">
      <Header health={health} onOpenSettings={openSettings} docsUrl={`${displayBaseUrl}/docs`} />
      <div className="layout">
        <main className="conversation" aria-label="Conversation">
          <MessageList turns={conversation.turns} onRate={conversation.rate} onPick={send} />
          <div>
            {conversation.retry ? (
              <div style={{ padding: "0 16px" }}>
                <div style={{ maxWidth: 820, margin: "0 auto 8px" }}>
                  <RetryBanner retry={conversation.retry} onCancel={conversation.stop} />
                </div>
              </div>
            ) : null}
            <Composer
              draft={draft}
              onDraft={setDraft}
              onSend={send}
              onStop={conversation.stop}
              busy={conversation.busy}
              disabled={false}
            />
          </div>
        </main>
        <div ref={sideRef} style={{ display: "contents" }}>
          <SidePanel
            tab={tab}
            onTab={setTab}
            settings={settings}
            onSettings={updateSettings}
            health={health}
            lastMeta={conversation.lastMeta}
            draft={draft}
            turnCount={conversation.turns.length}
            onClear={conversation.clear}
            authError={conversation.authError}
            onDismissAuthError={conversation.dismissAuthError}
            displayBaseUrl={displayBaseUrl}
          />
        </div>
      </div>
    </div>
  );
}
