import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble } from "./message-bubble";
import { Composer } from "./composer";
import { Scale, FileText, Search, Sparkles, BookOpen } from "lucide-react";
import type { ChatMessage } from "@/hooks/use-chat";

interface ChatPanelProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onSend: (message: string) => void;
  hasDocuments: boolean;
}

export function ChatPanel({
  messages,
  isLoading,
  onSend,
  hasDocuments,
}: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex h-full flex-col">
      <ScrollArea className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState hasDocuments={hasDocuments} onSend={onSend} />
        ) : (
          <div className="mx-auto max-w-3xl px-4 divide-y divide-border">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </ScrollArea>
      <Composer onSend={onSend} isLoading={isLoading} disabled={!hasDocuments} />
    </div>
  );
}

function EmptyState({
  hasDocuments,
  onSend,
}: {
  hasDocuments: boolean;
  onSend: (q: string) => void;
}) {
  const suggestions = [
    { icon: FileText, text: "Summarize the key points of the document" },
    { icon: Search, text: "What rights and obligations are described?" },
    { icon: BookOpen, text: "Explain the main articles in simple terms" },
    { icon: Sparkles, text: "List the most important provisions" },
  ];

  return (
    <div className="flex h-full min-h-[65vh] flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
        className="flex flex-col items-center gap-4 max-w-md"
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-bg-subtle">
          <Scale size={16} className="text-text-secondary" />
        </div>

        <div className="text-center">
          <h2 className="text-base font-medium text-text-bright">
            Legal AI
          </h2>
          <p className="mt-1 text-xs text-text-muted leading-relaxed">
            {hasDocuments
              ? "Your documents are ready. Ask anything."
              : "Upload legal documents to get started."}
          </p>
        </div>

        {hasDocuments && (
          <div className="mt-3 grid w-full grid-cols-1 gap-1.5 sm:grid-cols-2">
            {suggestions.map((s, i) => {
              const Icon = s.icon;
              return (
                <button
                  key={i}
                  onClick={() => onSend(s.text)}
                  className="group flex items-start gap-2 rounded-lg border border-border bg-bg-subtle p-3 text-left text-[12px] text-text-secondary transition-colors hover:border-border-hover hover:text-text cursor-pointer"
                >
                  <Icon size={12} className="mt-0.5 shrink-0 text-text-muted group-hover:text-text-secondary transition-colors" />
                  <span className="leading-snug">{s.text}</span>
                </button>
              );
            })}
          </div>
        )}
      </motion.div>
    </div>
  );
}
