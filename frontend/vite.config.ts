import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import path from 'path'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const backend = env.DEV_BACKEND_URL || 'http://127.0.0.1:8000';
  return {
    plugins: [
      react(),
      VitePWA({
        registerType: 'autoUpdate',
        // Extra static assets to precache alongside the app shell
        includeAssets: ['favicon.ico', 'apple-touch-icon-180x180.png', 'og-image.png'],
        manifest: {
          name: 'CRAG MultiHop RAG',
          short_name: 'CRAG RAG',
          description:
            'Chat with your PDFs, URLs and notes. Dense + BM25 retrieval, self-grading Corrective RAG, up to 3 query hops, local reranking and faithfulness scores on every answer.',
          theme_color: '#240091',
          background_color: '#ffffff',
          display: 'standalone',
          start_url: '/',
          scope: '/',
          icons: [
            {
              src: 'pwa-64x64.png',
              sizes: '64x64',
              type: 'image/png',
            },
            {
              src: 'pwa-192x192.png',
              sizes: '192x192',
              type: 'image/png',
            },
            {
              src: 'pwa-512x512.png',
              sizes: '512x512',
              type: 'image/png',
            },
            {
              src: 'maskable-icon-512x512.png',
              sizes: '512x512',
              type: 'image/png',
              purpose: 'maskable',
            },
          ],
        },
        workbox: {
          // Offline SPA shell: unknown routes fall back to index.html,
          // but never for the backend API / websocket endpoints.
          navigateFallback: '/index.html',
          navigateFallbackDenylist: [/^\/api\//, /^\/ws\//, /^\/admin/],
          runtimeCaching: [
            {
              // Fonts are the only cross-origin asset the app loads.
              urlPattern: /^https:\/\/fonts\.(googleapis|gstatic)\.com\//,
              handler: 'CacheFirst',
              options: {
                cacheName: 'google-fonts',
                expiration: {
                  maxEntries: 30,
                  maxAgeSeconds: 60 * 60 * 24 * 365,
                },
              },
            },
          ],
        },
      }),
    ],
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
