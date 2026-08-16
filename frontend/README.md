# DocIntel-AI — Frontend

A ChatGPT-styled single-page app for the DocIntel-AI backend. Built with Vite +
React 18 + TypeScript. Assistant answers render as Markdown (`react-markdown` +
`remark-gfm`) and stream in token-by-token over Server-Sent Events.

The design system is a light, achromatic "graphite ink on warm paper" look —
plain CSS custom properties in `src/styles/tokens.css`, no UI framework.

## Prerequisites

The backend must be running and reachable. From the repo root:

```bash
cd backend
uvicorn app.main:app --reload      # serves http://localhost:8000
```

Chat needs a `GOOGLE_API_KEY` in `backend/.env`. The first PDF upload downloads
the HuggingFace embedding model (`BAAI/bge-small-en-v1.5`), which can take a
moment.

## Run the frontend

```bash
cd frontend
npm install
cp .env.example .env        # optional; defaults to http://localhost:8000
npm run dev                 # http://localhost:5173
```

`npm run dev` starts Vite on port **5173** — the origin the backend's CORS
config already allows. Point `VITE_API_BASE` at a different backend URL in
`.env` if needed.

## Build & type-check

```bash
npm run build               # runs `tsc` then `vite build`
npm run preview             # serve the production build locally
```

## How it talks to the backend

All requests go directly to `VITE_API_BASE` (default `http://localhost:8000`),
under `/api`:

| Action        | Endpoint                              |
| ------------- | ------------------------------------- |
| Create session | `POST /api/sessions`                 |
| Status        | `GET /api/sessions/{id}`              |
| History       | `GET /api/sessions/{id}/messages`     |
| Upload PDFs   | `POST /api/sessions/{id}/documents`   |
| Chat (SSE)    | `POST /api/sessions/{id}/chat`        |
| Clear chat    | `POST /api/sessions/{id}/clear`       |
| Delete session | `DELETE /api/sessions/{id}`          |

The session id is persisted in `localStorage`. Because backend sessions are
in-memory, a backend restart invalidates a stored id — the app treats a `404`
as "start a fresh session and clear local history" automatically.

## Project layout

```
src/
├── api/
│   ├── client.ts      # one function per REST endpoint (typed errors)
│   └── stream.ts      # POST + SSE reader for the chat stream
├── hooks/
│   └── useChat.ts     # session/messages/status state machine
├── components/        # Sidebar, UploadPanel, StatusPanel, ChatHeader,
│   │                  # MessageList, MessageItem, SourceBadge, Composer,
│   │                  # EmptyState (+ co-located CSS)
├── styles/
│   ├── tokens.css     # the design-system custom properties
│   └── global.css     # reset, base type, themed scrollbar
├── types.ts           # shared domain types
├── App.tsx            # app shell (sidebar + conversation area)
└── main.tsx           # entry point
```
