import { useEffect, useRef } from 'react'

import type { ChatMessage } from '../types'
import { EmptyState } from './EmptyState'
import { MessageItem } from './MessageItem'
import './MessageList.css'

interface MessageListProps {
  messages: ChatMessage[]
  onPromptSelect: (text: string) => void
}

export function MessageList({ messages, onPromptSelect }: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const prevCount = useRef(0)

  // Stick to the bottom when a new message is added, and while streaming as long
  // as the user hasn't scrolled up to read earlier turns.
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const grew = messages.length > prevCount.current
    prevCount.current = messages.length
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 140
    if (grew || nearBottom) el.scrollTop = el.scrollHeight
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="messages messages--empty" ref={scrollRef}>
        <EmptyState onPromptSelect={onPromptSelect} />
      </div>
    )
  }

  return (
    <div className="messages" ref={scrollRef}>
      <div className="messages__inner">
        {messages.map((message, index) => (
          <MessageItem key={index} message={message} />
        ))}
      </div>
    </div>
  )
}
