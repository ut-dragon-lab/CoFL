import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { thirdPartyLicenses } from "./scripts/third-party-licenses";

export default defineConfig({
  plugins: [react(), thirdPartyLicenses()],
  server: { proxy: { "/api": "http://127.0.0.1:8787" } },
  build: {
    outDir: "../src/cofl/app/static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three", "@react-three/fiber", "@react-three/drei"],
          ui: ["@radix-ui/themes"],
        },
      },
    },
  },
});
