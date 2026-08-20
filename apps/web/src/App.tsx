import { Route, Routes } from 'react-router'

import { AppShell } from './components'
import {
  AlertsPage,
  CopilotPage,
  DriftPage,
  MachineDetailPage,
  MachinesPage,
  NotFoundPage,
  OverviewPage,
} from './pages'

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<OverviewPage />} />
        <Route path="machines" element={<MachinesPage />} />
        <Route path="machines/:machineCode" element={<MachineDetailPage />} />
        <Route path="alerts" element={<AlertsPage />} />
        <Route path="drift" element={<DriftPage />} />
        <Route path="copilot" element={<CopilotPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  )
}
