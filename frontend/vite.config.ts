import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server runs on :5173 to match the backend's default CORS origin
// (see backend/app/core/config.py -> cors_allow_origins).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
})
