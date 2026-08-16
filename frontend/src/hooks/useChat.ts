/**
 * useChat — owns the whole client-side conversation state machine:
 *   session id (persisted in localStorage), session status, messages, and the
 *   streaming lifecycle. All backend I/O funnels through here so components stay
 *   presentational.
 *
 * Resilience: backend sessions are in-memory, so a stored id goes stale (404)
 * after a restart. Any 404 on status/messages/chat/upload is treated as
 * "recreate a fresh session and clear local history".
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import {
  ApiError,
  SessionNotFound,
  clearChat,
  createSession,
  deleteSession,
  getMessages,
  getStatus,
  uploadDocuments,
} from '../api/client'
import { streamChat } from '../api/stream'
import type { ChatMessage, ProcessResult, SessionStatus } from '../types'

const STORAGE_KEY = 'docintel:session_id'

function readStoredId(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

function writeStoredId(id: string | null): void {
  try {
    if (id) window.localStorage.setItem(STORAGE_KEY, id)
    else window.localStorage.removeItem(STORAGE_KEY)
  } catch {
    /* localStorage unavailable (private mode) — session just won't persist */
  }
}

function errorMessage(err: unknown, fallback: string): string {
  if (err instanceof ApiError) return err.message
  if (err instanceof Error && err.message) return err.message
  return fallback
}

export interface UseChat {
  sessionId: string | null
  status: SessionStatus | null
  messages: ChatMessage[]
  isStreaming: boolean
  /** Stats from the most recent successful upload (pages/chunks); null until
   *  a document is processed in this session, and after a restart/new chat. */
  lastProcess: ProcessResult | null
  bootstrapping: boolean
  bootstrapError: string | null
  uploading: boolean
  uploadError: string | null
  send: (question: string) => void
  upload: (files: File[]) => Promise<void>
  clear: () => Promise<void>
  newSession: () => Promise<void>
  retry: () => void
  dismissUploadError: () => void
}

export function useChat(): UseChat {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [status, setStatus] = useState<SessionStatus | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [lastProcess, setLastProcess] = useState<ProcessResult | null>(null)
  const [bootstrapping, setBootstrapping] = useState(true)
  const [bootstrapError, setBootstrapError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  // Guards against React 18 StrictMode's double-invoked effect creating two
  // sessions on first mount.
  const bootstrappedRef = useRef(false)

  const persistSession = useCallback((id: string | null) => {
    setSessionId(id)
    writeStoredId(id)
  }, [])

  /** Create a brand-new session and reset local history/status. */
  const freshSession = useCallback(async (): Promise<string> => {
    const id = await createSession()
    persistSession(id)
    setMessages([])
    setLastProcess(null)
    try {
      setStatus(await getStatus(id))
    } catch {
      setStatus(null)
    }
    return id
  }, [persistSession])

  // -- Bootstrap on mount ---------------------------------------------------
  const bootstrap = useCallback(async () => {
    setBootstrapping(true)
    setBootstrapError(null)
    const stored = readStoredId()
    try {
      if (stored) {
        try {
          const [nextStatus, history] = await Promise.all([
            getStatus(stored),
            getMessages(stored),
          ])
          persistSession(stored)
          setStatus(nextStatus)
          setMessages(history)
          return
        } catch (err) {
          if (!(err instanceof SessionNotFound)) throw err
          // stale id -> fall through to a fresh session
          writeStoredId(null)
        }
      }
      await freshSession()
    } catch (err) {
      setBootstrapError(errorMessage(err, 'Could not start a session.'))
    } finally {
      setBootstrapping(false)
    }
  }, [freshSession, persistSession])

  useEffect(() => {
    if (bootstrappedRef.current) return
    bootstrappedRef.current = true
    void bootstrap()
    return () => abortRef.current?.abort()
  }, [bootstrap])

  // -- Streaming send -------------------------------------------------------
  const send = useCallback(
    (question: string) => {
      const trimmed = question.trim()
      if (!trimmed || isStreaming || !sessionId) return

      setMessages((prev) => [
        ...prev,
        { role: 'user', content: trimmed },
        { role: 'assistant', content: '', streaming: true },
      ])
      setIsStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      // Update the trailing assistant message (the streaming placeholder).
      const patchAssistant = (patch: Partial<ChatMessage>) => {
        setMessages((prev) => {
          if (prev.length === 0) return prev
          const next = prev.slice()
          const last = next[next.length - 1]
          if (last.role !== 'assistant') return prev
          next[next.length - 1] = { ...last, ...patch }
          return next
        })
      }

      streamChat(
        sessionId,
        trimmed,
        {
          onToken: (text) =>
            setMessages((prev) => {
              if (prev.length === 0) return prev
              const next = prev.slice()
              const last = next[next.length - 1]
              if (last.role !== 'assistant') return prev
              next[next.length - 1] = { ...last, content: last.content + text }
              return next
            }),
          onSources: (source) => patchAssistant({ source }),
          onError: (message) => patchAssistant({ error: message }),
          onDone: () => {
            patchAssistant({ streaming: false })
            setIsStreaming(false)
            abortRef.current = null
          },
        },
        controller.signal,
      ).catch(async (err) => {
        abortRef.current = null
        if ((err as { name?: string })?.name === 'AbortError') {
          setIsStreaming(false)
          return
        }
        if (err instanceof SessionNotFound) {
          // Session expired mid-request: reset and ask the user to resend.
          try {
            await freshSession()
          } catch {
            /* handled below via the note */
          }
          setMessages([
            {
              role: 'assistant',
              content: '',
              error:
                'Your previous session expired, so I started a new one. Please send your message again.',
            },
          ])
          setIsStreaming(false)
          return
        }
        patchAssistant({
          streaming: false,
          error: errorMessage(err, 'Sorry, I could not generate a response.'),
        })
        setIsStreaming(false)
      })
    },
    [freshSession, isStreaming, sessionId],
  )

  // -- Upload + process -----------------------------------------------------
  const upload = useCallback(
    async (files: File[]) => {
      if (!files.length || uploading || !sessionId) return
      setUploading(true)
      setUploadError(null)
      try {
        let id = sessionId
        let result: ProcessResult
        try {
          result = await uploadDocuments(id, files)
        } catch (err) {
          if (!(err instanceof SessionNotFound)) throw err
          id = await freshSession() // stale session -> recreate and retry once
          result = await uploadDocuments(id, files)
        }
        setLastProcess(result)
        // process_documents() resets server-side history; mirror that locally
        // and refresh the full status (pdf names + counts).
        setMessages([])
        try {
          setStatus(await getStatus(id))
        } catch {
          /* keep previous status if the refresh fails */
        }
      } catch (err) {
        setUploadError(errorMessage(err, 'Could not process the PDF(s).'))
      } finally {
        setUploading(false)
      }
    },
    [freshSession, sessionId, uploading],
  )

  // -- Clear chat -----------------------------------------------------------
  const clear = useCallback(async () => {
    if (!sessionId || isStreaming || messages.length === 0) return
    try {
      const nextStatus = await clearChat(sessionId)
      setMessages([])
      setStatus(nextStatus)
    } catch (err) {
      if (err instanceof SessionNotFound) {
        await freshSession().catch(() => undefined)
        return
      }
      // Non-fatal: surface nothing intrusive, just log to the console.
      console.error('Clear chat failed:', err)
    }
  }, [freshSession, isStreaming, messages.length, sessionId])

  // -- New chat (delete + recreate) -----------------------------------------
  const newSession = useCallback(async () => {
    if (isStreaming) return
    abortRef.current?.abort()
    const previous = sessionId
    setUploadError(null)
    // Best-effort delete of the old session; ignore failures (already gone, etc).
    if (previous) deleteSession(previous).catch(() => undefined)
    try {
      await freshSession()
    } catch (err) {
      setBootstrapError(errorMessage(err, 'Could not start a new session.'))
    }
  }, [freshSession, isStreaming, sessionId])

  const retry = useCallback(() => {
    void bootstrap()
  }, [bootstrap])

  const dismissUploadError = useCallback(() => setUploadError(null), [])

  return {
    sessionId,
    status,
    messages,
    isStreaming,
    lastProcess,
    bootstrapping,
    bootstrapError,
    uploading,
    uploadError,
    send,
    upload,
    clear,
    newSession,
    retry,
    dismissUploadError,
  }
}
