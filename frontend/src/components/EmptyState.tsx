import { IconChatDoc, IconQuote, IconRoute } from './icons'
import './EmptyState.css'

interface EmptyStateProps {
  onPromptSelect: (text: string) => void
}

const EXAMPLE_PROMPTS = [
  'Summarize the key takeaways from my document',
  'What is the difference between RAG and fine-tuning?',
  'Explain vector embeddings in simple terms',
]

const CAPABILITIES = [
  {
    Icon: IconChatDoc,
    title: 'Chat with your PDFs',
    body: 'Upload one or more PDFs and ask questions grounded in their contents.',
  },
  {
    Icon: IconRoute,
    title: 'Automatic RAG routing',
    body: 'Every question is scored against your documents to decide PDF vs. general knowledge.',
  },
  {
    Icon: IconQuote,
    title: 'Source attribution',
    body: 'Answers cite the file and page they came from — or mark general knowledge.',
  },
]

export function EmptyState({ onPromptSelect }: EmptyStateProps) {
  return (
    <div className="empty">
      <div className="empty__intro">
        <h1 className="empty__title">DocIntel-AI</h1>
        <p className="empty__subtitle">
          Ask anything. Upload PDFs to ground answers in your documents — otherwise
          I answer from general knowledge.
        </p>
      </div>

      <ul className="empty__prompts">
        {EXAMPLE_PROMPTS.map((prompt) => (
          <li key={prompt}>
            <button
              type="button"
              className="empty__prompt"
              onClick={() => onPromptSelect(prompt)}
            >
              {prompt}
            </button>
          </li>
        ))}
      </ul>

      <ul className="empty__cards">
        {CAPABILITIES.map(({ Icon, title, body }) => (
          <li key={title} className="empty__card">
            <span className="empty__card-icon">
              <Icon size={20} />
            </span>
            <h2 className="empty__card-title">{title}</h2>
            <p className="empty__card-body">{body}</p>
          </li>
        ))}
      </ul>
    </div>
  )
}
