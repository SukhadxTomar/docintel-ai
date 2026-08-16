import type { MessageSource } from '../types'
import { IconQuote } from './icons'
import './SourceBadge.css'

/**
 * The attribution row beneath an assistant answer.
 *  - rag: one achromatic pill per distinct source ("filename · p.N")
 *  - llm: a single "General AI Knowledge" chip
 */
export function SourceBadge({ source }: { source: MessageSource }) {
  if (source.type === 'llm') {
    return (
      <div className="source-badge" aria-label="Answer source">
        <span className="source-chip source-chip--general">{source.label}</span>
      </div>
    )
  }

  if (source.sources.length === 0) return null

  return (
    <div className="source-badge" aria-label="Sources">
      <span className="source-badge__label">Sources</span>
      <ul className="source-pills">
        {source.sources.map((ref, index) => (
          <li className="source-chip" key={`${ref.name}-${ref.page}-${index}`}>
            <IconQuote size={13} className="source-chip__icon" />
            <span className="source-chip__name">{ref.name}</span>
            {ref.page && ref.page !== 'Unknown' && (
              <span className="source-chip__page">· p.{ref.page}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
