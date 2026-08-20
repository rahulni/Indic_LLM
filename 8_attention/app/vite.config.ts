import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The site is published at the root of this repository's GitHub Pages site, which is
// served from https://rahulni.github.io/Indic_LLM/. Vite needs that prefix baked in or
// every asset URL resolves against the domain root and 404s.
//
// `npm run dev` and `npm run preview` serve from '/', so the base is applied only for
// the production build. Override with BASE_PATH if publishing somewhere else.
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === 'build' ? process.env.BASE_PATH ?? '/Indic_LLM/' : '/',
  build: {
    outDir: 'dist',
    // One bundle. The page must work as a single self-contained artifact, and at this
    // size code-splitting buys nothing but more requests.
    assetsInlineLimit: 4096,
  },
}))
