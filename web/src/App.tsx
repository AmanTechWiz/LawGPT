import { useState, useCallback } from "react";
import { Navbar } from "@/components/layout/navbar";
import { ChatPanel } from "@/components/chat/chat-panel";
import { UploadPanel } from "@/components/upload/upload-panel";
import { ToastProvider, useToast } from "@/components/ui/toast";
import { useChat } from "@/hooks/use-chat";
import { postSessionReset } from "@/lib/session-reset";

export default function App() {
  return (
    <ToastProvider>
      <AppContent />
    </ToastProvider>
  );
}

function AppContent() {
  const { addToast } = useToast();
  const [activeTab, setActiveTab] = useState<"chat" | "upload">("upload");
  const [documentCount, setDocumentCount] = useState(0);
  const [isResetting, setIsResetting] = useState(false);
  const { messages, isLoading, sendMessage, clearMessages } = useChat();

  const handleUploadSuccess = useCallback(() => {
    setDocumentCount((c) => c + 1);
    setActiveTab("chat");
  }, []);

  const handleResetServer = useCallback(async () => {
    if (
      !window.confirm(
        "Clear all uploaded documents, server cache (Redis), and this chat? This cannot be undone."
      )
    ) {
      return;
    }
    setIsResetting(true);
    try {
      const result = await postSessionReset();
      clearMessages();
      setDocumentCount(0);
      setActiveTab("upload");
      const n = result.truncated_tables?.length ?? 0;
      addToast({
        title: "Server reset complete",
        description:
          n > 0
            ? `Cleared ${n} table(s) and flushed cache. Upload documents again.`
            : "Cache flushed. If tables were empty, upload documents again.",
        variant: "success",
      });
    } catch (err) {
      addToast({
        title: "Reset failed",
        description: err instanceof Error ? err.message : "Could not reach the API.",
        variant: "destructive",
      });
    } finally {
      setIsResetting(false);
    }
  }, [addToast, clearMessages]);

  const hasDocuments = documentCount > 0;

  return (
    <div className="flex h-screen flex-col bg-background">
      <Navbar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        documentCount={documentCount}
        onResetServer={handleResetServer}
        isResetting={isResetting}
      />
      <main className="flex-1 overflow-hidden">
        {activeTab === "chat" ? (
          <ChatPanel
            messages={messages}
            isLoading={isLoading}
            onSend={sendMessage}
            hasDocuments={hasDocuments}
          />
        ) : (
          <UploadPanel
            onUploadSuccess={handleUploadSuccess}
            onClearServerData={handleResetServer}
            isClearingServer={isResetting}
          />
        )}
      </main>
    </div>
  );
}
