import { useRef, useState } from 'react'

import { ChatHeader } from './components/ChatHeader'
import { Composer } from './components/Composer'
import { IconAlert } from './components/icons'
import { MessageList } from './components/MessageList'
import { Sidebar } from './components/Sidebar'
import { useChat } from './hooks/useChat'
import { useTheme } from './hooks/useTheme'
import './App.css'

export default function App() {
  const chat = useChat()
  const { theme, toggleTheme } = useTheme()
  const [draft, setDraft] = useState('')
  const composerRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = () => {
    const text = draft.trim()
    if (!text) return
    chat.send(text)
    setDraft('')
  }

  const handlePromptSelect = (text: string) => {
    setDraft(text)
    // Focus after the value is applied so the caret lands at the end.
    requestAnimationFrame(() => composerRef.current?.focus())
  }

  const composerDisabled = chat.isStreaming || !chat.sessionId
  // Only block the canvas when we truly have no working session to fall back on.
  const showBlocker = !!chat.bootstrapError && !chat.sessionId

  return (
    <div className="app">
      <Sidebar
        status={chat.status}
        lastProcess={chat.lastProcess}
        messageCount={chat.messages.length}
        isStreaming={chat.isStreaming}
        uploading={chat.uploading}
        uploadError={chat.uploadError}
        onNewChat={() => void chat.newSession()}
        onUpload={(files) => chat.upload(files)}
        onClear={() => void chat.clear()}
        onDismissUploadError={chat.dismissUploadError}
      />

      <main className="conversation">
        <ChatHeader status={chat.status} theme={theme} onToggleTheme={toggleTheme} />

        {showBlocker ? (
          <div className="conversation__blocker">
            <div className="conversation__note" role="alert">
              <IconAlert size={18} className="conversation__note-icon" />
              <div className="conversation__note-body">
                <p>{chat.bootstrapError}</p>
                <button
                  type="button"
                  className="conversation__retry"
                  onClick={chat.retry}
                >
                  Try again
                </button>
              </div>
            </div>
          </div>
        ) : (
          <MessageList messages={chat.messages} onPromptSelect={handlePromptSelect} />
        )}

        <Composer
          value={draft}
          onChange={setDraft}
          onSubmit={handleSubmit}
          disabled={composerDisabled}
          textareaRef={composerRef}
        />
      </main>
    </div>
  )
}
