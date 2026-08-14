import { ApiError } from '../api/client'

interface ErrorStateProps {
  title: string
  error: unknown
  onRetry?: () => void
}

export function ErrorState({ title, error, onRetry }: ErrorStateProps) {
  return (
    <div className="state-panel state-panel--error" role="alert">
      <div>
        <h2>{title}</h2>
        <p>{getErrorMessage(error)}</p>
      </div>
      {onRetry ? (
        <button type="button" className="button button--primary" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  )
}

function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 404) {
      return 'The requested resource was not found.'
    }

    return error.message
  }

  if (error instanceof Error) {
    return error.message
  }

  return 'The API could not be reached.'
}

