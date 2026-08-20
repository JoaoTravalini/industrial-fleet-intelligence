import { lazy, Suspense } from 'react'
import type { ComponentType, ReactNode } from 'react'
import { Route, Routes } from 'react-router'

import { AppShell } from './components/AppShell'
import { LoadingState } from './components/LoadingState'

const OverviewPage = lazy(() => import('./pages/OverviewPage').then(namedPage('OverviewPage')))
const MachinesPage = lazy(() => import('./pages/MachinesPage').then(namedPage('MachinesPage')))
const MachineDetailPage = lazy(() =>
  import('./pages/MachineDetailPage').then(namedPage('MachineDetailPage')),
)
const AlertsPage = lazy(() => import('./pages/AlertsPage').then(namedPage('AlertsPage')))
const DriftPage = lazy(() => import('./pages/DriftPage').then(namedPage('DriftPage')))
const CopilotPage = lazy(() => import('./pages/CopilotPage').then(namedPage('CopilotPage')))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage').then(namedPage('NotFoundPage')))

type PageModule<TName extends string> = Record<TName, ComponentType>

function namedPage<TName extends string>(name: TName) {
  return (module: PageModule<TName>) => ({ default: module[name] })
}

function lazyRoute(element: ReactNode) {
  return <Suspense fallback={<LoadingState message="Loading dashboard route" />}>{element}</Suspense>
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={lazyRoute(<OverviewPage />)} />
        <Route path="machines" element={lazyRoute(<MachinesPage />)} />
        <Route path="machines/:machineCode" element={lazyRoute(<MachineDetailPage />)} />
        <Route path="alerts" element={lazyRoute(<AlertsPage />)} />
        <Route path="drift" element={lazyRoute(<DriftPage />)} />
        <Route path="copilot" element={lazyRoute(<CopilotPage />)} />
        <Route path="*" element={lazyRoute(<NotFoundPage />)} />
      </Route>
    </Routes>
  )
}
