import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

class TestResizeObserver implements ResizeObserver {
  constructor(_callback?: ResizeObserverCallback) {}

  observe(_target: Element) {}
  unobserve(_target: Element) {}
  disconnect() {}
}

globalThis.ResizeObserver = TestResizeObserver as typeof ResizeObserver

afterEach(() => {
  cleanup()
})
