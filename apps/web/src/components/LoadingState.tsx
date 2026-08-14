interface LoadingStateProps {
  message: string
}

export function LoadingState({ message }: LoadingStateProps) {
  return (
    <div className="state-panel" role="status" aria-live="polite">
      <div className="loading-indicator" aria-hidden="true" />
      <p>{message}</p>
    </div>
  )
}
