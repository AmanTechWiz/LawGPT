import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { uploadPDFs } from "@/lib/api";
import {
  Upload,
  FileText,
  X,
  Check,
  Loader2,
  CloudUpload,
  Trash2,
} from "lucide-react";

interface UploadPanelProps {
  onUploadSuccess: () => void;
  onClearServerData?: () => void | Promise<void>;
  isClearingServer?: boolean;
}

interface QueuedFile {
  id: string;
  file: File;
  status: "pending" | "uploading" | "done" | "error";
  error?: string;
}

export function UploadPanel({
  onUploadSuccess,
  onClearServerData,
  isClearingServer = false,
}: UploadPanelProps) {
  const [files, setFiles] = useState<QueuedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { addToast } = useToast();

  const addFiles = useCallback(
    (newFiles: FileList | File[]) => {
      const pdfFiles = Array.from(newFiles).filter(
        (f) => f.type === "application/pdf" || f.name.endsWith(".pdf")
      );
      if (pdfFiles.length === 0) {
        addToast({ title: "Only PDF files are supported", variant: "destructive" });
        return;
      }
      setFiles((prev) => [
        ...prev,
        ...pdfFiles.map((file) => ({
          id: crypto.randomUUID(),
          file,
          status: "pending" as const,
        })),
      ]);
    },
    [addToast]
  );

  const removeFile = (id: string) =>
    setFiles((prev) => prev.filter((f) => f.id !== id));

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      addFiles(e.dataTransfer.files);
    },
    [addFiles]
  );

  const handleUpload = async () => {
    const pending = files.filter((f) => f.status === "pending");
    if (pending.length === 0) return;

    setIsUploading(true);
    setFiles((prev) =>
      prev.map((f) =>
        f.status === "pending" ? { ...f, status: "uploading" as const } : f
      )
    );

    try {
      const result = await uploadPDFs(pending.map((f) => f.file));
      setFiles((prev) =>
        prev.map((f) =>
          f.status === "uploading" ? { ...f, status: "done" as const } : f
        )
      );
      addToast({
        title: "Documents uploaded",
        description: result.message ?? `${pending.length} file(s) ingested.`,
        variant: "success",
      });
      onUploadSuccess();
    } catch (err) {
      setFiles((prev) =>
        prev.map((f) =>
          f.status === "uploading"
            ? { ...f, status: "error" as const, error: err instanceof Error ? err.message : "Failed" }
            : f
        )
      );
      addToast({
        title: "Upload failed",
        description: err instanceof Error ? err.message : "Something went wrong",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const pendingCount = files.filter((f) => f.status === "pending").length;

  return (
    <div className="mx-auto flex h-full max-w-lg flex-col items-center justify-center px-4 py-12">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.25 }}
        className="w-full"
      >
        <div className="mb-6 text-center">
          <h2 className="text-base font-medium text-text-bright">
            Upload Documents
          </h2>
          <p className="mt-1 text-xs text-text-muted">
            Add PDF files to your knowledge base
          </p>
        </div>

        {onClearServerData && (
          <div className="mb-5 w-full rounded-lg border border-red-500/25 bg-red-500/[0.07] p-3">
            <p className="mb-2 text-center text-[11px] leading-snug text-red-200/85">
              Wipe Postgres documents, Redis, and server memory cache so answers are recomputed from scratch.
            </p>
            <Button
              type="button"
              variant="destructive"
              size="lg"
              disabled={isClearingServer}
              onClick={() => onClearServerData()}
              className="h-10 w-full border border-red-500/30 text-sm font-semibold"
            >
              {isClearingServer ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Clearing…
                </>
              ) : (
                <>
                  <Trash2 size={16} />
                  Clear database &amp; cache
                </>
              )}
            </Button>
          </div>
        )}

        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border border-dashed p-10 transition-colors ${
            isDragOver
              ? "border-text-secondary bg-accent-muted"
              : "border-border hover:border-border-hover"
          }`}
        >
          <CloudUpload size={20} className="text-text-muted" />
          <div className="text-center">
            <p className="text-xs font-medium text-text-secondary">
              Drop PDF files here
            </p>
            <p className="mt-0.5 text-[10px] text-text-muted">or click to browse</p>
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files);
            e.target.value = "";
          }}
          className="hidden"
        />

        <AnimatePresence>
          {files.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-3 space-y-1.5"
            >
              {files.map((f) => (
                <motion.div
                  key={f.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center gap-2.5 rounded-lg border border-border bg-bg-subtle px-3 py-2"
                >
                  <FileText size={13} className="shrink-0 text-text-muted" />
                  <div className="flex-1 min-w-0">
                    <p className="truncate text-xs font-medium text-text">
                      {f.file.name}
                    </p>
                    <p className="text-[10px] text-text-muted">
                      {(f.file.size / 1024).toFixed(0)} KB
                      {f.status === "error" && (
                        <span className="ml-1 text-red-400">— {f.error}</span>
                      )}
                    </p>
                  </div>
                  {f.status === "uploading" && (
                    <Loader2 size={13} className="animate-spin text-text-secondary" />
                  )}
                  {f.status === "done" && (
                    <Check size={13} className="text-text-secondary" />
                  )}
                  {(f.status === "pending" || f.status === "error") && (
                    <button
                      onClick={(e) => { e.stopPropagation(); removeFile(f.id); }}
                      className="text-text-muted hover:text-text-secondary cursor-pointer"
                    >
                      <X size={13} />
                    </button>
                  )}
                </motion.div>
              ))}

              {pendingCount > 0 && (
                <Button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className="mt-2 w-full"
                >
                  {isUploading ? (
                    <><Loader2 size={13} className="animate-spin" /> Processing...</>
                  ) : (
                    <><Upload size={13} /> Upload {pendingCount} file{pendingCount > 1 ? "s" : ""}</>
                  )}
                </Button>
              )}

              {pendingCount === 0 && files.every((f) => f.status === "done") && (
                <p className="pt-1 text-center text-[10px] text-text-muted">
                  All files processed — switch to Chat
                </p>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
