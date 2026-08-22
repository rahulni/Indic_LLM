import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages on this repository serves the committed branch contents directly, so a
// file at 8_attention/index.html appears at
// https://rahulni.github.io/Indic_LLM/8_attention/ - the same pattern the earlier
// submissions use. Vite needs that full prefix baked in, or every asset URL resolves
// against the domain root and 404s.
//
// `npm run dev` serves from '/', so the base applies only to the production build.
// Override with BASE_PATH if publishing somewhere else.
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === 'build' ? process.env.BASE_PATH ?? '/Indic_LLM/8_attention/' : '/',
  build: {
    outDir: 'dist',
    // One bundle. The page must work as a single self-contained artifact, and at this
    // size code-splitting buys nothing but more requests.
    assetsInlineLimit: 4096,
  },
}))
