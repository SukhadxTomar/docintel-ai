import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent,
} from 'react'

import { IconAlert, IconClose, IconFile, IconUpload } from './icons'
import './UploadPanel.css'

interface UploadPanelProps {
  onUpload: (files: File[]) => void
  uploading: boolean
  uploadError: string | null
  onDismissError: () => void
  /** Disabled while a chat turn is streaming. */
  disabled?: boolean
}

function isPdf(file: File): boolean {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
}

export function UploadPanel({
  onUpload,
  uploading,
  uploadError,
  onDismissError,
  disabled = false,
}: UploadPanelProps) {
  const [pending, setPending] = useState<File[]>([])
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const wasUploading = useRef(false)

  // Clear the staged files only once a process run has completed cleanly.
  useEffect(() => {
    if (wasUploading.current && !uploading && !uploadError) setPending([])
    wasUploading.current = uploading
  }, [uploading, uploadError])

  const busy = disabled || uploading

  const addFiles = (fileList: FileList | null) => {
    if (!fileList) return
    const incoming = Array.from(fileList).filter(isPdf)
    if (incoming.length === 0) return
    setPending((prev) => {
      const seen = new Set(prev.map((file) => `${file.name}:${file.size}`))
      const merged = prev.slice()
      for (const file of incoming) {
        const key = `${file.name}:${file.size}`
        if (!seen.has(key)) {
          seen.add(key)
          merged.push(file)
        }
      }
      return merged
    })
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    if (busy) return
    addFiles(event.dataTransfer.files)
  }

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    if (!busy) setDragActive(true)
  }

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
  }

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    addFiles(event.target.files)
    event.target.value = '' // allow re-picking the same file
  }

  const openPicker = () => {
    if (!busy) inputRef.current?.click()
  }

  const handleZoneKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openPicker()
    }
  }

  const canProcess = pending.length > 0 && !busy

  return (
    <section className="upload" aria-label="Upload PDFs">
      <div
        className={
          'upload__drop' +
          (dragActive ? ' upload__drop--active' : '') +
          (busy ? ' upload__drop--disabled' : '')
        }
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={openPicker}
        onKeyDown={handleZoneKeyDown}
        role="button"
        tabIndex={0}
        aria-disabled={busy}
      >
        <IconUpload size={20} className="upload__drop-icon" />
        <p className="upload__drop-text">
          <span className="upload__drop-strong">Drag &amp; drop PDFs</span>
          <br />
          or click to browse
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          className="upload__input"
          onChange={handleInputChange}
          tabIndex={-1}
        />
      </div>

      {pending.length > 0 && (
        <ul className="upload__pending">
          {pending.map((file, index) => (
            <li className="upload__pending-item" key={`${file.name}-${index}`}>
              <IconFile size={15} className="upload__pending-icon" />
              <span className="upload__pending-name" title={file.name}>
                {file.name}
              </span>
              <button
                type="button"
                className="upload__pending-remove"
                onClick={() => setPending((prev) => prev.filter((_, i) => i !== index))}
                disabled={uploading}
                aria-label={`Remove ${file.name}`}
              >
                <IconClose size={14} />
              </button>
            </li>
          ))}
        </ul>
      )}

      <button
        type="button"
        className="upload__process"
        onClick={() => onUpload(pending)}
        disabled={!canProcess}
      >
        {uploading ? (
          <>
            <span className="upload__spinner" aria-hidden="true" />
            Processing…
          </>
        ) : pending.length > 0 ? (
          `Process ${pending.length} PDF${pending.length > 1 ? 's' : ''}`
        ) : (
          'Process'
        )}
      </button>

      {uploadError && (
        <div className="upload__error" role="alert">
          <IconAlert size={15} className="upload__error-icon" />
          <span className="upload__error-text">{uploadError}</span>
          <button
            type="button"
            className="upload__error-dismiss"
            onClick={onDismissError}
            aria-label="Dismiss error"
          >
            <IconClose size={14} />
          </button>
        </div>
      )}
    </section>
  )
}
