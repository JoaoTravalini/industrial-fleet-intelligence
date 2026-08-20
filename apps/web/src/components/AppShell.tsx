import { NavLink, Outlet } from 'react-router'

import { useHealthStatus } from '../api/queries'

const navigationItems = [
  { label: 'Overview', path: '/', end: true },
  { label: 'Machines', path: '/machines' },
  { label: 'Alerts', path: '/alerts' },
  { label: 'Drift Monitoring', path: '/drift' },
  { label: 'AI Copilot', path: '/copilot' },
]

export function AppShell() {
  const health = useHealthStatus()
  const connectionLabel = health.isLoading
    ? 'Checking API'
    : health.isError
      ? 'API unavailable'
      : 'API connected'
  const connectionTone = health.isError ? 'danger' : health.isLoading ? 'neutral' : 'success'

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Application sidebar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            IF
          </div>
          <div>
            <p className="brand-title">Industrial Fleet Intelligence</p>
            <p className="brand-subtitle">Local Portfolio Platform</p>
          </div>
        </div>

        <nav className="primary-nav" aria-label="Primary navigation">
          {navigationItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.end}
              className={({ isActive }) => (isActive ? 'nav-link is-active' : 'nav-link')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className={`connection-dot connection-dot--${connectionTone}`} aria-hidden="true" />
          <span aria-live="polite">{connectionLabel}</span>
        </div>
      </aside>

      <div className="content-shell">
        <header className="mobile-header">
          <div>
            <p className="brand-title">Industrial Fleet Intelligence</p>
            <p className="brand-subtitle">Local Portfolio Platform</p>
          </div>
          <span className={`connection-pill connection-pill--${connectionTone}`} aria-live="polite">
            {connectionLabel}
          </span>
        </header>

        <nav className="mobile-nav" aria-label="Primary navigation mobile">
          {navigationItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.end}
              className={({ isActive }) => (isActive ? 'nav-link is-active' : 'nav-link')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <main className="main-content" id="main-content">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
