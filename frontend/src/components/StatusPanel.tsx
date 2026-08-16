import type { SessionStatus } from '../types'
import './StatusPanel.css'

export function StatusPanel({ status }: { status: SessionStatus | null }) {
  const pdfLoaded = !!status?.pdf_loaded
  const documentCount = status?.pdf_names.length ?? 0
  const mode = pdfLoaded ? 'Document chat' : 'General chat'

  return (
    <section className="status" aria-label="Session status">
      <h2 className="status__heading">Status</h2>
      <dl className="status__rows">
        <div className="status__row">
          <dt>PDF loaded</dt>
          <dd>{pdfLoaded ? 'Yes' : 'No'}</dd>
        </div>
        {documentCount > 0 && (
          <div className="status__row">
            <dt>Documents</dt>
            <dd>{documentCount}</dd>
          </div>
        )}
        <div className="status__row">
          <dt>Mode</dt>
          <dd>
            <span className="status__badge">{mode}</span>
          </dd>
        </div>
      </dl>
      {!pdfLoaded && (
        <p className="status__hint">General chat mode — answers come from general AI knowledge.</p>
      )}
    </section>
  )
}
