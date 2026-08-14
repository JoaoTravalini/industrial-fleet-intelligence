import { formatInteger } from '../utils/format'

interface PaginationProps {
  count: number
  limit: number
  offset: number
  total: number
  onPageChange: (nextPage: number) => void
}

export function Pagination({ count, limit, offset, total, onPageChange }: PaginationProps) {
  const currentPage = Math.floor(offset / limit) + 1
  const firstRecord = total === 0 ? 0 : offset + 1
  const lastRecord = total === 0 ? 0 : offset + count
  const hasPrevious = offset > 0
  const hasNext = offset + count < total

  return (
    <div className="pagination" aria-label="Pagination controls">
      <p className="pagination-range">
        Showing {formatInteger(firstRecord)}-{formatInteger(lastRecord)} of {formatInteger(total)}
      </p>
      <div className="pagination-actions">
        <button
          type="button"
          className="button button--secondary"
          onClick={() => onPageChange(currentPage - 1)}
          disabled={!hasPrevious}
          aria-label="Previous page"
        >
          Previous
        </button>
        <span className="page-number" aria-label={`Current page ${currentPage}`}>
          Page {currentPage}
        </span>
        <button
          type="button"
          className="button button--secondary"
          onClick={() => onPageChange(currentPage + 1)}
          disabled={!hasNext}
          aria-label="Next page"
        >
          Next
        </button>
      </div>
    </div>
  )
}
