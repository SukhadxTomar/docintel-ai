import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

/** localStorage key. Must match the bootstrap script in index.html. */
const STORAGE_KEY = 'docintel-theme'

/**
 * Resolve the theme for React's initial state. The inline bootstrap script in
 * index.html already resolved and applied a theme (stored choice → OS
 * preference) before React mounted, so we trust the `data-theme` attribute it
 * set — that keeps state in lockstep with the already-painted DOM. The
 * localStorage / matchMedia fallbacks only matter if that script didn't run.
 */
function getInitialTheme(): Theme {
  const attr = document.documentElement.getAttribute('data-theme')
  if (attr === 'light' || attr === 'dark') return attr

  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    /* localStorage unavailable (private mode, etc.) — fall through */
  }

  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

/** Manage the light/dark theme: reflect it onto <html data-theme> and persist. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* ignore persistence failures — the in-memory theme still applies */
    }
  }, [theme])

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))
  }, [])

  return { theme, toggleTheme }
}
