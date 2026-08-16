import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import type { ChatMessage } from '../types'
import { IconAlert } from './icons'
import { SourceBadge } from './SourceBadge'
import './MessageItem.css'

export function MessageItem({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="msg msg--user">
        <div className="msg__bubble">{message.content}</div>
      </div>
    )
  }

  const showTyping = !!message.streaming && message.content.length === 0 && !message.error
  const showCaret = !!message.streaming && message.content.length > 0

  return (
    <div className="msg msg--assistant">
      <div className="msg__body">
        {message.content.length > 0 && (
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        )}

        {showCaret && <span className="msg__caret" aria-hidden="true" />}

        {showTyping && (
          <div className="msg__typing" role="status" aria-label="Generating response">
            <span />
            <span />
            <span />
          </div>
        )}

        {message.error && (
          <div className="msg__error" role="status">
            <IconAlert size={16} className="msg__error-icon" />
            <span>{message.error}</span>
          </div>
        )}

        {message.source && !message.streaming && <SourceBadge source={message.source} />}
      </div>
    </div>
  )
}
