import type { ProcessResult, SessionStatus } from '../types'
import { IconBrand, IconCompose, IconFile, IconTrash } from './icons'
import { StatusPanel } from './StatusPanel'
import { UploadPanel } from './UploadPanel'
import './Sidebar.css'

interface SidebarProps {
  status: SessionStatus | null
  lastProcess: ProcessResult | null
  messageCount: number
  isStreaming: boolean
  uploading: boolean
  uploadError: string | null
  onNewChat: () => void
  onUpload: (files: File[]) => void
  onClear: () => void
  onDismissUploadError: () => void
}

export function Sidebar({
  status,
  lastProcess,
  messageCount,
  isStreaming,
  uploading,
  uploadError,
  onNewChat,
  onUpload,
  onClear,
  onDismissUploadError,
}: SidebarProps) {
  const pdfNames = status?.pdf_names ?? []
  const hasPdfs = pdfNames.length > 0
  const canClear = messageCount > 0 && !isStreaming

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <IconBrand size={22} className="sidebar__brand-mark" />
        <span className="sidebar__brand-name">DocIntel-AI</span>
      </div>

      <button
        type="button"
        className="sidebar__new"
        onClick={onNewChat}
        disabled={isStreaming}
      >
        <IconCompose size={17} />
        <span>New chat</span>
      </button>

      <div className="sidebar__scroll">
        <UploadPanel
          onUpload={onUpload}
          uploading={uploading}
          uploadError={uploadError}
          onDismissError={onDismissUploadError}
          disabled={isStreaming}
        />

        {!hasPdfs && (
          <p className="sidebar__hint">
            Upload PDFs to ground answers in your documents.
          </p>
        )}

        {hasPdfs && (
          <section className="sidebar__pdfs" aria-label="Uploaded PDFs">
            <h2 className="sidebar__section-title">Uploaded PDFs</h2>
            <ul className="sidebar__pdf-list">
              {pdfNames.map((name, index) => (
                <li className="sidebar__pdf" key={`${name}-${index}`}>
                  <IconFile size={15} className="sidebar__pdf-icon" />
                  <span className="sidebar__pdf-name" title={name}>
                    {name}
                  </span>
                </li>
              ))}
            </ul>
            <p className="sidebar__pdf-meta">
              {pdfNames.length} file{pdfNames.length > 1 ? 's' : ''}
              {lastProcess
                ? ` · ${lastProcess.document_count} pages · ${lastProcess.chunk_count} chunks`
                : ''}
            </p>
          </section>
        )}

        <StatusPanel status={status} />
      </div>

      <div className="sidebar__footer">
        <button
          type="button"
          className="sidebar__clear"
          onClick={onClear}
          disabled={!canClear}
        >
          <IconTrash size={16} />
          <span>Clear chat</span>
        </button>
        <p className="sidebar__foot-note">Hybrid RAG · Gemini + FAISS</p>
      </div>
    </aside>
  )
}
