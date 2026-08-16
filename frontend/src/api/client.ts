/**
 * Typed REST client for the DocIntel-AI backend. One function per endpoint.
 * The chat stream lives in ./stream.ts (it needs raw stream access, not JSON).
 *
 * All calls go directly to VITE_API_BASE (CORS is configured on the backend).
 */
import type {
  ChatMessage,
  MessageSource,
  ProcessResult,
  RagSourceRef,
  Role,
  SessionStatus,
} from '../types'

export const API_BASE = (
  import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'
).replace(/\/+$/, '')

/** Any non-OK HTTP response (or a network failure, with status 0). */
export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** A 404 — the session no longer exists (backend restarted / expired). The UI
 *  treats this as "recreate a fresh session and clear local history". */
export class SessionNotFound extends ApiError {
  constructor(message = 'Session not found.') {
    super(message, 404)
    this.name = 'SessionNotFound'
  }
}

const NETWORK_MESSAGE = `Cannot reach the server at ${API_BASE}. Is the backend running?`

/** Build an ApiError/SessionNotFound from a non-OK response, pulling FastAPI's
 *  `detail` (a string for HTTPException, or a list for 422 validation errors). */
async function toError(res: Response): Promise<ApiError> {
  let detail = res.statusText || `Request failed (${res.status}).`
  try {
    const data = await res.json()
    if (typeof data?.detail === 'string') {
      detail = data.detail
    } else if (Array.isArray(data?.detail) && typeof data.detail[0]?.msg === 'string') {
      detail = data.detail[0].msg
    }
  } catch {
    /* body was not JSON — keep the status-based message */
  }
  return res.status === 404 ? new SessionNotFound(detail) : new ApiError(detail, res.status)
}

/** fetch wrapper: turns network failures into a friendly ApiError(status 0) and
 *  non-OK responses into typed errors. */
async function request(path: string, init?: RequestInit): Promise<Response> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new ApiError(NETWORK_MESSAGE, 0)
  }
  if (!res.ok) throw await toError(res)
  return res
}

/** Coerce a raw RAG `sources` array into typed refs (shared with stream.ts). */
export function toRagRefs(raw: unknown): RagSourceRef[] {
  if (!Array.isArray(raw)) return []
  return raw
    .filter((s): s is Record<string, unknown> => !!s && typeof s === 'object')
    .map((s) => ({
      name: String(s.name ?? 'Unknown'),
      page: String(s.page ?? 'Unknown'),
    }))
}

function normalizeStoredSource(raw: unknown): MessageSource | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const source = raw as Record<string, unknown>
  if (source.type === 'rag') {
    return { type: 'rag', sources: toRagRefs(source.sources) }
  }
  if (source.type === 'llm') {
    return {
      type: 'llm',
      label: typeof source.label === 'string' ? source.label : 'General AI Knowledge',
    }
  }
  return undefined
}

function normalizeRole(role: unknown): Role {
  return role === 'user' ? 'user' : 'assistant'
}

// -- Endpoints --------------------------------------------------------------

/** POST /api/sessions -> { session_id } */
export async function createSession(): Promise<string> {
  const res = await request('/api/sessions', { method: 'POST' })
  const data = await res.json()
  return String(data.session_id)
}

/** GET /api/sessions/{id} -> status */
export async function getStatus(sessionId: string): Promise<SessionStatus> {
  const res = await request(`/api/sessions/${sessionId}`)
  return (await res.json()) as SessionStatus
}

/** GET /api/sessions/{id}/messages -> normalized ChatMessage[] */
export async function getMessages(sessionId: string): Promise<ChatMessage[]> {
  const res = await request(`/api/sessions/${sessionId}/messages`)
  const data = await res.json()
  const messages: unknown[] = Array.isArray(data?.messages) ? data.messages : []
  return messages.map((m) => {
    const message = (m ?? {}) as Record<string, unknown>
    return {
      role: normalizeRole(message.role),
      content: typeof message.content === 'string' ? message.content : '',
      source: normalizeStoredSource(message.source),
    }
  })
}

/** POST /api/sessions/{id}/documents (multipart, repeatable `files`) */
export async function uploadDocuments(
  sessionId: string,
  files: File[],
): Promise<ProcessResult> {
  const form = new FormData()
  for (const file of files) form.append('files', file, file.name)
  // No explicit Content-Type: the browser sets the multipart boundary.
  const res = await request(`/api/sessions/${sessionId}/documents`, {
    method: 'POST',
    body: form,
  })
  return (await res.json()) as ProcessResult
}

/** POST /api/sessions/{id}/clear -> status */
export async function clearChat(sessionId: string): Promise<SessionStatus> {
  const res = await request(`/api/sessions/${sessionId}/clear`, { method: 'POST' })
  return (await res.json()) as SessionStatus
}

/** DELETE /api/sessions/{id} (204) */
export async function deleteSession(sessionId: string): Promise<void> {
  await request(`/api/sessions/${sessionId}`, { method: 'DELETE' })
}
