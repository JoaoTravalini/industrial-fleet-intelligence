import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'

import { ApiError } from '../api/client'
import { sendCopilotMessage, useCopilotHealth } from '../api/queries'
import type { CopilotChatResponse, CopilotHistoryMessage, CopilotSource } from '../api/types'
import { ErrorState, LoadingState, PageHeader, Section, StatusBadge } from '../components'

const MAX_MESSAGE_LENGTH = 4000
const MAX_HISTORY_MESSAGES = 6
const suggestedQuestions = [
  'Fleet overview',
  'Explain MCH-0001',
  'What does anomaly score mean?',
  'Explain current drift',
  'Why did the latest MCH-0001 prediction receive that model output?',
]

type ConversationMessage =
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string; sources: CopilotSource[]; model: string }

type CopilotError = {
  title: string
  message: string
}

export function CopilotPage() {
  const health = useCopilotHealth()
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [draft, setDraft] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [errorState, setErrorState] = useState<CopilotError | null>(null)

  useEffect(() => {
    if (!isSending) {
      setElapsedSeconds(0)
      return undefined
    }

    const startedAt = Date.now()
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [isSending])

  const isUnavailable = health.data?.status === 'unavailable' || health.isError
  const statusLabel = health.isLoading
    ? 'Checking local AI'
    : isUnavailable
      ? 'Local AI unavailable'
      : 'Local AI available'

  async function submitMessage(messageText: string) {
    const message = messageText.trim()
    if (!message || isSending || message.length > MAX_MESSAGE_LENGTH) {
      return
    }

    setErrorState(null)
    setIsSending(true)
    const history: CopilotHistoryMessage[] = messages
      .slice(-MAX_HISTORY_MESSAGES)
      .map((item) => ({ role: item.role, content: item.content }))
    setMessages((current) => [...current, { role: 'user', content: message }])
    setDraft('')

    try {
      const response = await sendCopilotMessage({ message, history })
      setMessages((current) => [...current, assistantMessage(response)])
    } catch (error) {
      setErrorState(copilotErrorFrom(error))
    } finally {
      setIsSending(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void submitMessage(draft)
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Local AI"
        title="AI Copilot"
        description="Local, read-only assistant for operational platform data."
      />

      <Section
        title="Copilot Session"
        description="Runs locally with Ollama and uses read-only, source-grounded platform evidence."
      >
        <div className="copilot-status-row">
          <StatusBadge value={isUnavailable ? 'critical' : health.isLoading ? 'checking' : 'active'} />
          <strong>{statusLabel}</strong>
          <span>{health.data?.model ?? 'qwen3:4b-instruct'}</span>
          <span>{health.data?.model_loaded ? 'Model loaded' : 'Model loads on demand'}</span>
          <span>Runs locally with Ollama</span>
          <span>Copilot is read-only and cannot change platform state.</span>
        </div>

        {isUnavailable ? (
          <div className="copilot-unavailable" role="status">
            Start Ollama and ensure qwen3:4b-instruct is installed.
          </div>
        ) : null}

        {messages.length === 0 ? (
          <div className="suggested-question-grid" aria-label="Suggested questions">
            {suggestedQuestions.map((question) => (
              <button
                className="suggested-question"
                key={question}
                type="button"
                disabled={isSending}
                onClick={() => void submitMessage(question)}
              >
                {question}
              </button>
            ))}
          </div>
        ) : null}

        <div className="chat-thread" aria-live="polite">
          {messages.map((message, index) => (
            <article className={`chat-message chat-message--${message.role}`} key={`${message.role}-${index}`}>
              <p className="chat-role">{message.role === 'user' ? 'You' : 'AI Copilot'}</p>
              <p className="chat-content">{message.content}</p>
              {message.role === 'assistant' ? <SourceList sources={message.sources} model={message.model} /> : null}
            </article>
          ))}
          {isSending ? (
            <LoadingState
              message={`Local model is processing. Local inference can take longer on the first request. ${elapsedSeconds}s elapsed.`}
            />
          ) : null}
          {errorState ? <ErrorState title={errorState.title} error={new Error(errorState.message)} /> : null}
        </div>

        <form className="chat-composer" onSubmit={handleSubmit}>
          <label className="field-control">
            <span>Message</span>
            <textarea
              aria-label="Copilot message"
              maxLength={MAX_MESSAGE_LENGTH}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask about fleet state, machine predictions, anomaly scores, drift, alerts, or SHAP attribution."
              value={draft}
            />
          </label>
          <div className="chat-actions">
            <button className="button button--secondary" type="button" onClick={() => setMessages([])}>
              Clear conversation
            </button>
            <button className="button button--primary" disabled={!draft.trim() || isSending} type="submit">
              Send
            </button>
          </div>
        </form>
      </Section>
    </div>
  )
}

function copilotErrorFrom(error: unknown): CopilotError {
  if (error instanceof ApiError && error.status === 504) {
    return {
      title: 'Local model response timed out',
      message: 'Verify Ollama is running, then try again after the model is warm.',
    }
  }

  return {
    title: 'Copilot unavailable',
    message: 'Local AI Copilot is unavailable. Start Ollama and try again.',
  }
}

function assistantMessage(response: CopilotChatResponse): ConversationMessage {
  return {
    role: 'assistant',
    content: response.answer,
    sources: response.sources,
    model: response.model,
  }
}

function SourceList({ sources, model }: { sources: CopilotSource[]; model: string }) {
  return (
    <div className="source-list">
      <p>Sources</p>
      <ul>
        {sources.map((source) => (
          <li key={`${source.type}-${source.id}`}>{source.label}</li>
        ))}
      </ul>
      <small>Model: {model}</small>
    </div>
  )
}
