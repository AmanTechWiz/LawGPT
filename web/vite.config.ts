import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/ingest-pdfs": "http://localhost:8001",
      "/stream": "http://localhost:8001",
      "/query": "http://localhost:8001",
      "/health": "http://localhost:8001",
      "/session": "http://localhost:8001",
    },
  },
});
