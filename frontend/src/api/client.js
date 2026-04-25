/**
 * Axios API client for the Condition Monitoring System backend.
 * Base URL is proxied through Vite dev server in development.
 */

import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

const client = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 30_000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Request interceptor (add auth token when needed) ──────────────────────
client.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error),
)

// ── Response interceptor (normalise errors) ───────────────────────────────
client.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'Unknown API error'
    return Promise.reject(new Error(message))
  },
)

// ── API methods ───────────────────────────────────────────────────────────

export const api = {
  /** Classify fault from a 3-axis vibration window. */
  classify: (payload) => client.post('/analysis/classify', payload).then((r) => r.data),

  /** Extract feature vector (optionally includes antigravity features). */
  features: (payload) => client.post('/analysis/features', payload).then((r) => r.data),

  /** Dedicated antigravity regime analysis. */
  antigravity: (payload) => client.post('/analysis/antigravity', payload).then((r) => r.data),

  /** Get all model metadata. */
  models: () => client.get('/models/').then((r) => r.data),

  /** Health check. */
  health: () => client.get('/health', { baseURL: API_BASE }).then((r) => r.data),
}

export default client
