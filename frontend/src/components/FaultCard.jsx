/**
 * FaultCard — Displays the primary fault classification result.
 *
 * Shows:
 *   - Predicted fault class with colour-coded badge
 *   - Confidence percentage with animated fill bar
 *   - Per-class probability distribution (mini bars)
 *   - ⚠ Reduced Gravity Regime Detected badge (amber) when ag_prob > 0.30
 */

import React from 'react'
import useStore from '../store/useStore'

const FAULT_COLOURS = {
  normal:       { bg: '#22c55e', label: '✅ Normal',       badge: 'badge-green' },
  bpfi:         { bg: '#3b82f6', label: '🔵 BPFI', badge: 'badge-blue' },
  bpfo:         { bg: '#06b6d4', label: '🔵 BPFO', badge: 'badge-blue' },
  bsf:          { bg: '#8b5cf6', label: '🟣 BSF',  badge: 'badge-purple' },
  unbalance:    { bg: '#f59e0b', label: '🟡 Unbalance',    badge: 'badge-amber' },
  misalignment: { bg: '#f97316', label: '🟠 Misalignment', badge: 'badge-amber' },
  looseness:    { bg: '#ec4899', label: '🩷 Looseness',    badge: 'badge-amber' },
  antigravity:  { bg: '#4B0082', label: '⚛ Antigravity',  badge: 'badge-purple' },
}

function ProbBar({ label, value, colour }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-muted" style={{ textTransform: 'capitalize', fontWeight: 500 }}>{label}</span>
        <span className="text-xs text-mono" style={{ color: 'var(--text-primary)' }}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="gauge-track" style={{ height: 5 }}>
        <div
          className="gauge-fill"
          style={{ width: `${value * 100}%`, background: colour, opacity: 0.85 }}
        />
      </div>
    </div>
  )
}

export default function FaultCard() {
  const classification = useStore((s) => s.classification)
  const antigravityProbability = useStore((s) => s.antigravityProbability)
  const antigravityEnabled = useStore((s) => s.antigravityEnabled)

  const showReducedGravity = antigravityEnabled &&
    antigravityProbability !== null &&
    antigravityProbability > 0.30

  if (!classification) {
    return (
      <div className="glass-card p-6 fade-in" style={{ textAlign: 'center' }}>
        <span style={{ fontSize: '2.5rem' }}>⚙️</span>
        <p className="text-muted mt-4">No analysis run yet.</p>
        <p className="text-xs text-dimmed mt-2">Upload a signal or run a demo analysis.</p>
      </div>
    )
  }

  const { predicted_class, confidence, class_probabilities } = classification
  const meta = FAULT_COLOURS[predicted_class] || FAULT_COLOURS.normal

  return (
    <div className="glass-card p-6 fade-in">
      {/* Primary verdict */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <p className="text-xs text-dimmed font-medium" style={{ letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 6 }}>
            Fault Classification
          </p>
          <h2 className="text-2xl font-bold" style={{ color: meta.bg }}>{meta.label}</h2>
        </div>
        <div style={{ textAlign: 'right' }}>
          <p className="text-3xl font-bold text-mono" style={{ color: 'var(--text-primary)' }}>
            {(confidence * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-dimmed">confidence</p>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="gauge-track mb-6" style={{ height: 10 }}>
        <div
          className="gauge-fill"
          style={{
            width: `${confidence * 100}%`,
            background: `linear-gradient(90deg, ${meta.bg}88, ${meta.bg})`,
            boxShadow: `0 0 12px ${meta.bg}55`,
          }}
        />
      </div>

      {/* Reduced gravity badge */}
      {showReducedGravity && (
        <div
          className="badge badge-amber fade-in"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginBottom: 16,
            padding: '8px 14px',
            borderRadius: 10,
            fontSize: '0.8rem',
            background: 'rgba(245,158,11,0.12)',
            border: '1px solid rgba(245,158,11,0.3)',
            animation: 'none',
            fontWeight: 500,
            letterSpacing: 'normal',
            textTransform: 'none',
          }}
        >
          ⚠️ Reduced Gravity Regime Detected
          <span className="text-mono" style={{ fontSize: '0.75rem', opacity: 0.8 }}>
            ({(antigravityProbability * 100).toFixed(1)}%)
          </span>
        </div>
      )}

      {/* Per-class probability bars */}
      <div>
        <p className="text-xs text-dimmed font-medium mb-4" style={{ textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Class Probabilities
        </p>
        {Object.entries(class_probabilities)
          .sort(([, a], [, b]) => b - a)
          .map(([cls, prob]) => (
            <ProbBar
              key={cls}
              label={cls}
              value={prob}
              colour={FAULT_COLOURS[cls]?.bg || '#6b7280'}
            />
          ))}
      </div>

      <div className="flex justify-between mt-6" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 14 }}>
        <span className="text-xs text-dimmed">Session: <code className="text-mono" style={{ color: 'var(--text-code)' }}>{classification.session_id}</code></span>
        <span className="text-xs text-dimmed">Window #{classification.window_index}</span>
      </div>
    </div>
  )
}
