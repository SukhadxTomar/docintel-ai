import type { SessionStatus } from '../types'
import type { Theme } from '../hooks/useTheme'
import { IconMoon, IconSun } from './icons'
import './ChatHeader.css'

interface ChatHeaderProps {
  status: SessionStatus | null
  theme: Theme
  onToggleTheme: () => void
}

export function ChatHeader({ status, theme, onToggleTheme }: ChatHeaderProps) {
  const pdfLoaded = !!status?.pdf_loaded
  const title = pdfLoaded ? 'Document chat' : 'General chat'
  const caption = pdfLoaded
    ? 'Grounded in your uploaded PDFs, with automatic fallback to general knowledge.'
    : 'Answering from general knowledge — upload PDFs in the sidebar to ground answers in your documents.'

  const isDark = theme === 'dark'
  const toggleLabel = isDark ? 'Switch to light mode' : 'Switch to dark mode'

  return (
    <header className="chat-header">
      <div className="chat-header__inner">
        <div className="chat-header__text">
          <h1 className="chat-header__title">{title}</h1>
          <p className="chat-header__caption">{caption}</p>
        </div>
        <button
          type="button"
          className="chat-header__theme"
          onClick={onToggleTheme}
          aria-label={toggleLabel}
          title={toggleLabel}
        >
          {isDark ? <IconSun size={18} /> : <IconMoon size={18} />}
        </button>
      </div>
    </header>
  )
}
