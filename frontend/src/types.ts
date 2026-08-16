/** Shared domain types, mirroring the backend contract (backend/app/api). */

export type Role = 'user' | 'assistant'

/** A single distinct citation from a RAG answer. `page` is a label string
 *  ("1", "12", or "Unknown") — see backend util `page_label`. */
export interface RagSourceRef {
  name: string
  page: string
}

/**
 * Where an assistant answer came from. This is the *unified* shape used both by
 * the stored history messages (GET .../messages -> message.source) and by the
 * normalized SSE `sources` event (see api/stream.ts).
 */
export type MessageSource =
  | { type: 'rag'; sources: RagSourceRef[] }
  | { type: 'llm'; label: string }

export interface ChatMessage {
  role: Role
  content: string
  source?: MessageSource
  /** True while assistant tokens are still streaming in. */
  streaming?: boolean
  /** Set when this turn failed; rendered as an achromatic note (no red). */
  error?: string
}

/** GET /api/sessions/{id} */
export interface SessionStatus {
  session_id: string
  pdf_loaded: boolean
  chat_ready: boolean
  document_count: number
  pdf_names: string[]
  processing_done: boolean
  message_count: number
}

/** POST /api/sessions/{id}/documents */
export interface ProcessResult {
  session_id: string
  pdf_names: string[]
  document_count: number
  chunk_count: number
  processing_done: boolean
}
