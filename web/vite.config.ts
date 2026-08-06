import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    // three.js is one indivisible ~1 MB vendor chunk; the warning is meant to
    // catch bloated APP code, and the app chunk is ~60 kB after the split
    chunkSizeWarningLimit: 1100,
    // No manualChunks. There used to be one naming three/fiber/drei and katex
    // as vendor chunks, and it actively prevented the split it looked like it
    // was making: forcing @react-three/fiber into a chunk dragged React in
    // with it, the entry statically needs React, so the 1.1 MB "vendor" chunk
    // was modulepreloaded on every page including the ones with no 3-D at all.
    // CloudView and PhysicsBody are dynamic imports now, so Rollup derives the
    // same chunks from real reachability and they load only when entered.
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
