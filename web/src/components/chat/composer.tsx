import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ArrowUp, Square } from "lucide-react";

interface ComposerProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export function Composer({ onSend, isLoading, disabled }: ComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = `${Math.min(ta.scrollHeight, 150)}px`;
    }
  }, [value]);

  const handleSubmit = () => {
    if (!value.trim() || isLoading || disabled) return;
    onSend(value);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-border bg-bg/80 backdrop-blur-xl px-4 py-3">
      <div className="mx-auto max-w-3xl">
        <div
          className={`flex items-end gap-2 rounded-lg border p-1.5 transition-colors ${
            disabled
              ? "border-border opacity-40 pointer-events-none"
              : "border-border-hover focus-within:border-text-muted"
          }`}
        >
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              disabled
                ? "Upload documents first..."
                : "Ask a legal question..."
            }
            disabled={disabled}
            rows={1}
            className="flex-1 resize-none border-0 bg-transparent px-2 py-1.5 text-[13px] text-text placeholder:text-text-muted focus:outline-none"
          />
          <Button
            size="icon"
            variant={value.trim() && !isLoading ? "default" : "ghost"}
            onClick={handleSubmit}
            disabled={!value.trim() || isLoading || disabled}
            className="h-7 w-7 shrink-0 rounded-md"
          >
            {isLoading ? <Square size={12} /> : <ArrowUp size={14} />}
          </Button>
        </div>
        <p className="mt-1.5 text-center text-[10px] text-text-muted">
          Answers are grounded in your uploaded documents
        </p>
      </div>
    </div>
  );
}
