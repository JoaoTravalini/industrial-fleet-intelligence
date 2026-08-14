import { Link } from 'react-router'

import { PageHeader } from '../components'

export function NotFoundPage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Not Found"
        title="Page not found"
        description="The requested dashboard route does not exist."
        actions={
          <Link className="button button--primary" to="/">
            Return to Overview
          </Link>
        }
      />
    </div>
  )
}
