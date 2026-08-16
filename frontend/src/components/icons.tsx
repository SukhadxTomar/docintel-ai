/**
 * Minimal, achromatic line icons. All stroke `currentColor` on no fill, so they
 * inherit ink color and stay consistent with the graphite design system.
 */
import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function Icon({ size = 18, children, ...props }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {children}
    </svg>
  )
}

/** Brand mark: a document with a subtle spark — "document intelligence". */
export function IconBrand(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M6 3.5h7l5 5V19a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 6 19V5A1.5 1.5 0 0 1 7.5 3.5Z" />
      <path d="M13 3.5V8.5H18" />
      <path d="M9.4 14.2l.7 1.6 1.6.7-1.6.7-.7 1.6-.7-1.6-1.6-.7 1.6-.7.7-1.6Z" />
    </Icon>
  )
}

/** New chat (compose). */
export function IconCompose(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 20h8" />
      <path d="M15.5 4.5a2.12 2.12 0 0 1 3 3L8 18l-4 1 1-4Z" />
    </Icon>
  )
}

export function IconUpload(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 15V4" />
      <path d="M8 8l4-4 4 4" />
      <path d="M5 15v2.5A1.5 1.5 0 0 0 6.5 19h11a1.5 1.5 0 0 0 1.5-1.5V15" />
    </Icon>
  )
}

export function IconFile(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M7 3.5h6l4 4V19a1.5 1.5 0 0 1-1.5 1.5h-8.5A1.5 1.5 0 0 1 5.5 19V5A1.5 1.5 0 0 1 7 3.5Z" />
      <path d="M13 3.5V7.5H17" />
    </Icon>
  )
}

export function IconSend(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 19V5" />
      <path d="M6 11l6-6 6 6" />
    </Icon>
  )
}

export function IconTrash(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 6.5h16" />
      <path d="M9 6.5V5A1.5 1.5 0 0 1 10.5 3.5h3A1.5 1.5 0 0 1 15 5v1.5" />
      <path d="M6.5 6.5 7.3 19a1.5 1.5 0 0 0 1.5 1.4h6.4a1.5 1.5 0 0 0 1.5-1.4l.8-12.5" />
    </Icon>
  )
}

/** Small "!" note glyph for achromatic error/info notes. */
export function IconAlert(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 8v4.5" />
      <path d="M12 15.6h.01" />
    </Icon>
  )
}

export function IconClose(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M6 6l12 12" />
      <path d="M18 6 6 18" />
    </Icon>
  )
}

/** Capability: chat with your PDFs. */
export function IconChatDoc(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v9A1.5 1.5 0 0 1 18.5 16H9l-4 3.5V16H5.5A1.5 1.5 0 0 1 4 14.5Z" />
      <path d="M8 8.5h8" />
      <path d="M8 11.5h5" />
    </Icon>
  )
}

/** Capability: automatic routing (a fork in the path). */
export function IconRoute(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="6" cy="18" r="2" />
      <circle cx="18" cy="6" r="2" />
      <circle cx="18" cy="18" r="2" />
      <path d="M6 16V9a3 3 0 0 1 3-3h7" />
      <path d="M18 8v8" />
    </Icon>
  )
}

/** Capability: source attribution (a quote mark). */
export function IconQuote(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M9 7H6a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1v1a2 2 0 0 1-2 2" />
      <path d="M19 7h-3a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1v1a2 2 0 0 1-2 2" />
    </Icon>
  )
}
