import { useEffect, type KeyboardEvent, type RefObject } from 'react'

import { IconSend } from './icons'
import './Composer.css'

const MAX_HEIGHT = 200

interface ComposerProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  disabled?: boolean
  placeholder?: string
  textareaRef: RefObject<HTMLTextAreaElement>
}

/** Auto-growing chat input. Enter sends, Shift+Enter inserts a newline. */
export function Composer({
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder,
  textareaRef,
}: ComposerProps) {
  // Grow with content up to MAX_HEIGHT, then scroll internally.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`
  }, [value, textareaRef])

  const canSend = !disabled && value.trim().length > 0

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      if (canSend) onSubmit()
    }
  }

  return (
    <div className="composer">
      <div className={`composer__field${disabled ? ' composer__field--disabled' : ''}`}>
        <textarea
          ref={textareaRef}
          className="composer__textarea"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder ?? 'Message DocIntel-AI…'}
          rows={1}
          aria-label="Message"
        />
        <button
          type="button"
          className="composer__send"
          onClick={() => {
            if (canSend) onSubmit()
          }}
          disabled={!canSend}
          aria-label="Send message"
          title="Send"
        >
          <IconSend size={18} />
        </button>
      </div>
      <p className="composer__hint">
        Each question is routed to your PDFs or general knowledge. Verify important info.
      </p>
    </div>
  )
}
