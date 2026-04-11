import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { resetSessionIfEnabled } from "@/lib/session-reset";

const root = document.getElementById("root")!;

resetSessionIfEnabled()
  .catch((err) => console.warn("Session reset skipped or failed:", err))
  .finally(() => {
    createRoot(root).render(
      <StrictMode>
        <App />
      </StrictMode>
    );
  });
