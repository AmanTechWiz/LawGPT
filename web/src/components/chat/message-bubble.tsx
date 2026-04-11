import { motion } from "framer-motion";
import { User, Scale } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "@/hooks/use-chat";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isStreaming = message.role === "assistant" && message.isStreaming;
  const isEmpty = !message.content && message.role === "assistant";

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.15 }}
      className={`flex gap-3 py-4 ${isUser ? "flex-row-reverse" : ""}`}
    >
      <div
        className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full mt-0.5 ${
          isUser ? "bg-bg-elevated" : "border border-border bg-bg-subtle"
        }`}
      >
        {isUser ? (
          <User size={11} className="text-text-secondary" />
        ) : (
          <Scale size={10} className="text-text-secondary" />
        )}
      </div>

      <div className={`min-w-0 max-w-[85%] ${isUser ? "text-right" : ""}`}>
        <p className="mb-1 text-[10px] font-medium uppercase tracking-widest text-text-muted">
          {isUser ? "You" : "Legal AI"}
        </p>

        {isEmpty ? (
          <div className="flex flex-col gap-2 pt-0.5">
            <div className="h-3 w-56 animate-pulse rounded bg-bg-elevated" />
            <div className="h-3 w-40 animate-pulse rounded bg-bg-elevated" />
            <div className="h-3 w-28 animate-pulse rounded bg-bg-elevated" />
          </div>
        ) : isUser ? (
          <div className="inline-block rounded-xl rounded-tr-sm bg-bg-elevated px-3.5 py-2 text-left">
            <p className="text-[13px] leading-relaxed text-text whitespace-pre-wrap">
              {message.content}
            </p>
          </div>
        ) : (
          <div className="text-[13px] leading-[1.8] text-text">
            <div className="prose max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
            {isStreaming && <Cursor />}
          </div>
        )}

        {message.metadata?.citations &&
          message.metadata.citations.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {message.metadata.citations.map((c, i) => (
                <span
                  key={i}
                  className="rounded bg-accent-muted px-1.5 py-0.5 text-[10px] text-text-muted"
                >
                  {c}
                </span>
              ))}
            </div>
          )}
      </div>
    </motion.div>
  );
}

function Cursor() {
  return (
    <motion.span
      className="ml-0.5 inline-block h-4 w-[2px] rounded-full bg-text-secondary align-middle"
      animate={{ opacity: [1, 0] }}
      transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse" }}
    />
  );
}
