import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const backend = env.DEV_BACKEND_URL || 'http://127.0.0.1:8000';
  return {
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/ws': { target: backend, changeOrigin: true, ws: true },
    },
  },
  };
})
