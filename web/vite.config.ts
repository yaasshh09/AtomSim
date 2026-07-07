import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    // three.js is one indivisible ~1 MB vendor chunk; the warning is meant to
    // catch bloated APP code, and the app chunk is ~60 kB after the split
    chunkSizeWarningLimit: 1100,
    rollupOptions: {
      output: {
        // vendor split: three.js and katex dwarf the app code; separate chunks
        // cache independently (react rides with fiber inside the three chunk)
        manualChunks: {
          three: ["three", "@react-three/fiber", "@react-three/drei"],
          katex: ["katex"],
        },
      },
    },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
