import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { StatusBadge } from './StatusBadge'

describe('StatusBadge', () => {
  it('renders drift status with explicit text semantics', () => {
    render(<StatusBadge kind="drift" value="drift" />)

    expect(screen.getByText('Drift')).toHaveClass('status-badge--danger')
  })

  it('renders model decisions without calling them failures', () => {
    render(<StatusBadge kind="decision" value={true} />)

    expect(screen.getByText('Positive')).toHaveClass('status-badge--warning')
  })

  it('renders anomaly flags as flags rather than probabilities', () => {
    render(<StatusBadge kind="flag" value={false} />)

    expect(screen.getByText('Not flagged')).toHaveClass('status-badge--success')
  })
})
