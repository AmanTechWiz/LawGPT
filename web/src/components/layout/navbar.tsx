import { Scale, MessageSquare, Upload, Trash2, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface NavbarProps {
  activeTab: "chat" | "upload";
  onTabChange: (tab: "chat" | "upload") => void;
  documentCount: number;
  onResetServer: () => void | Promise<void>;
  isResetting: boolean;
}

export function Navbar({
  activeTab,
  onTabChange,
  documentCount,
  onResetServer,
  isResetting,
}: NavbarProps) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-bg/95 backdrop-blur-xl">
      <div className="mx-auto w-full max-w-3xl px-3 py-2 sm:flex sm:h-14 sm:items-center sm:justify-between sm:py-0 sm:px-4">
        {/* Row 1 (mobile): brand + tabs — never clips the reset control off-screen */}
        <div className="flex items-center justify-between gap-2 sm:contents">
          <div className="flex min-w-0 items-center gap-2">
            <Scale size={16} className="shrink-0 text-text-secondary" />
            <span className="truncate text-sm font-medium text-text">Legal AI</span>
          </div>

          <nav className="flex shrink-0 items-center gap-0.5 rounded-lg border border-border bg-bg-subtle p-0.5 sm:order-none">
            <TabButton
              active={activeTab === "chat"}
              onClick={() => onTabChange("chat")}
            >
              <MessageSquare size={13} />
              Chat
            </TabButton>
            <TabButton
              active={activeTab === "upload"}
              onClick={() => onTabChange("upload")}
            >
              <Upload size={13} />
              Upload
              {documentCount > 0 && (
                <span className="ml-0.5 flex h-4 min-w-4 items-center justify-center rounded bg-accent-muted px-1 text-[10px] font-medium text-text-secondary">
                  {documentCount}
                </span>
              )}
            </TabButton>
          </nav>
        </div>

        {/* Full-width reset row on mobile; inline on sm+ */}
        <button
          type="button"
          title="Clear database, Redis cache, and chat"
          disabled={isResetting}
          onClick={() => onResetServer()}
          className={cn(
            "mt-2 flex h-9 w-full items-center justify-center gap-2 rounded-md border border-red-500/35 bg-red-500/10 px-3 text-xs font-semibold text-red-300 transition-colors",
            "hover:border-red-400/50 hover:bg-red-500/20",
            "disabled:pointer-events-none disabled:opacity-50",
            "sm:mt-0 sm:h-8 sm:w-auto sm:shrink-0 sm:px-2.5"
          )}
        >
          {isResetting ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <Trash2 size={14} />
          )}
          Clear DB &amp; cache
        </button>
      </div>
    </header>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors cursor-pointer",
        active
          ? "bg-bg-elevated text-text-bright"
          : "text-text-muted hover:text-text-secondary"
      )}
    >
      {children}
    </button>
  );
}
