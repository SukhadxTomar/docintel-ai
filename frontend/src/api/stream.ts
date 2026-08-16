/**
 * Chat streaming over Server-Sent Events.
 *
 * The chat endpoint is a POST that responds with `text/event-stream`, so the
 * browser's EventSource (GET-only) can't be used. Instead we POST with fetch and
 * read `response.body` ourselves, splitting the byte stream into SSE frames on
 * blank lines and JSON-parsing each `data:` payload.
 *
 * Backend event shapes (see backend/app/api/routes/chat.py):
 *   {"type":"token","text": "..."}
 *   {"type":"sources","mode":"rag","sources":[{name,page}]}
 *   {"type":"sources","mode":"llm","sources":[],"label":"General AI Knowledge"}
 *   {"type":"error","message":"..."}
 *   {"type":"done","request_id":"..."}   <- always last
 *
 * The `sources` event carries `mode`; we normalize it to the unified
 * MessageSource shape ({type:'rag'|'llm', ...}) used everywhere in the UI.
 */
import { API_BASE, ApiError, SessionNotFound, toRagRefs } from './client'
import type { MessageSource } from '../types'

export interface StreamHandlers {
  onToken: (text: string) => void
  onSources: (source: MessageSource) => void
  onError: (message: string) => void
  /** Fired exactly once when the stream ends (server `done`, or stream close). */
  onDone: (requestId?: string) => void
}

function normalizeSourcesEvent(event: Record<string, unknown>): MessageSource {
  if (event.mode === 'rag') {
    return { type: 'rag', sources: toRagRefs(event.sources) }
  }
  return {
    type: 'llm',
    label: typeof event.label === 'string' ? event.label : 'General AI Knowledge',
  }
}

function dispatch(event: Record<string, unknown>, handlers: StreamHandlers): void {
  switch (event.type) {
    case 'token':
      handlers.onToken(typeof event.text === 'string' ? event.text : '')
      break
    case 'sources':
      handlers.onSources(normalizeSourcesEvent(event))
      break
    case 'error':
      handlers.onError(
        typeof event.message === 'string'
          ? event.message
          : 'The server reported an error.',
      )
      break
    case 'done':
      handlers.onDone(
        typeof event.request_id === 'string' ? event.request_id : undefined,
      )
      break
    default:
      break
  }
}

/** Parse one SSE frame (which may span multiple `data:` lines) and dispatch it. */
function handleFrame(rawFrame: string, handlers: StreamHandlers): void {
  const dataLines: string[] = []
  for (const line of rawFrame.split('\n')) {
    const clean = line.replace(/\r$/, '')
    if (clean.startsWith('data:')) {
      dataLines.push(clean.slice(5).replace(/^ /, ''))
    }
    // `:`-comments and other SSE fields (event:, id:, retry:) are ignored.
  }
  if (dataLines.length === 0) return

  const payload = dataLines.join('\n')
  if (payload === '[DONE]') {
    handlers.onDone()
    return
  }

  let event: unknown
  try {
    event = JSON.parse(payload)
  } catch {
    return // ignore a malformed frame rather than tearing down the stream
  }
  if (event && typeof event === 'object') {
    dispatch(event as Record<string, unknown>, handlers)
  }
}

/**
 * Open the chat stream and drive the handlers until it closes.
 *
 * Throws `SessionNotFound` on 404 (caller should recreate the session),
 * `ApiError` on other transport/HTTP failures, or an `AbortError` if `signal`
 * fired. Errors *inside* the stream arrive via `onError`, not by throwing.
 */
export async function streamChat(
  sessionId: string,
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}/api/sessions/${sessionId}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ question }),
      signal,
    })
  } catch (err) {
    if ((err as { name?: string })?.name === 'AbortError') throw err
    throw new ApiError(
      `Cannot reach the server at ${API_BASE}. Is the backend running?`,
      0,
    )
  }

  if (!res.ok) {
    if (res.status === 404) throw new SessionNotFound()
    let detail = `Chat request failed (${res.status}).`
    try {
      const data = await res.json()
      if (typeof data?.detail === 'string') detail = data.detail
    } catch {
      /* keep status-based message */
    }
    throw new ApiError(detail, res.status)
  }

  if (!res.body) {
    throw new ApiError('The server returned an empty response stream.', res.status)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let sep = buffer.indexOf('\n\n')
      while (sep !== -1) {
        const frame = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)
        handleFrame(frame, handlers)
        sep = buffer.indexOf('\n\n')
      }
    }
    // Flush a trailing frame that wasn't terminated by a blank line.
    const tail = buffer + decoder.decode()
    if (tail.trim()) handleFrame(tail, handlers)
  } finally {
    reader.releaseLock()
  }
}
