/**
 * SeverityGauge — Visual severity legend with all 8 fault/regime classes.
 * Antigravity is shown in deep purple (#4B0082) distinct from all other zones.
 */

import React from 'react'
import useStore from '../store/useStore'

const FAULT_LEGEND = [
  { key: 'normal',       colour: '#22c55e', label: 'Normal',        severity: 0 },
  { key: 'bpfi',         colour: '#3b82f6', label: 'BPFI',          severity: 2 },
  { key: 'bpfo',         colour: '#06b6d4', label: 'BPFO',          severity: 2 },
  { key: 'bsf',          colour: '#8b5cf6', label: 'BSF',           severity: 3 },
  { key: 'unbalance',    colour: '#f59e0b', label: 'Unbalance',     severity: 2 },
  { key: 'misalignment', colour: '#f97316', label: 'Misalignment',  severity: 3 },
  { key: 'looseness',    colour: '#ec4899', label: 'Looseness',     severity: 3 },
  { key: 'antigravity',  colour: '#4B0082', label: 'Antigravity',   severity: null, experimental: true },
]

const SEVERITY_LABELS = {
  0: { label: 'Healthy',  colour: '#22c55e' },
  1: { label: 'Watch',    colour: '#84cc16' },
  2: { label: 'Moderate', colour: '#f59e0b' },
  3: { label: 'Severe',   colour: '#ef4444' },
}

export default function SeverityGauge() {
  const predicted = useStore((s) => s.classification?.predicted_class)
  const antigravityEnabled = useStore((s) => s.antigravityEnabled)

  return (
    <div className="glass-card p-4">
      <p className="text-xs text-dimmed font-medium mb-4" style={{ textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        Fault Class Legend
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {FAULT_LEGEND
          .filter((f) => f.key !== 'antigravity' || antigravityEnabled)
          .map((item) => {
            const isActive = predicted === item.key
            const sevMeta = item.severity !== null ? SEVERITY_LABELS[item.severity] : null
            return (
              <div
                key={item.key}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '7px 10px',
                  borderRadius: 8,
                  background: isActive ? `${item.colour}18` : 'transparent',
                  border: isActive ? `1px solid ${item.colour}55` : '1px solid transparent',
                  transition: 'all 0.25s ease',
                  position: 'relative',
                }}
              >
                {/* Colour dot */}
                <div style={{
                  width: 10, height: 10, borderRadius: '50%',
                  background: item.colour,
                  boxShadow: isActive ? `0 0 8px ${item.colour}99` : 'none',
                  flexShrink: 0,
                }} />

                {/* Label */}
                <span className="text-sm" style={{
                  color: isActive ? item.colour : 'var(--text-secondary)',
                  fontWeight: isActive ? 600 : 400,
                  flex: 1,
                }}>
                  {item.label}
                  {item.experimental && (
                    <span style={{ fontSize: '0.65rem', marginLeft: 6, color: '#a855f7', opacity: 0.8 }}>
                      [experimental]
                    </span>
                  )}
                </span>

                {/* Severity rating */}
                {sevMeta ? (
                  <span style={{ fontSize: '0.7rem', color: sevMeta.colour, fontWeight: 500 }}>
                    {sevMeta.label}
                  </span>
                ) : (
                  <span style={{ fontSize: '0.7rem', color: '#a855f7', fontWeight: 500 }}>
                    Regime
                  </span>
                )}

                {/* Active indicator */}
                {isActive && (
                  <div style={{
                    position: 'absolute',
                    right: 10,
                    width: 6, height: 6,
                    borderRadius: '50%',
                    background: item.colour,
                    boxShadow: `0 0 8px ${item.colour}`,
                    animation: 'pulse-ring 1.5s ease-out infinite',
                  }} />
                )}
              </div>
            )
          })}
      </div>

      {/* Severity scale */}
      <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border-subtle)' }}>
        <p className="text-xs text-dimmed mb-3" style={{ textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 500 }}>
          Severity Scale
        </p>
        <div style={{ display: 'flex', gap: 0, borderRadius: 8, overflow: 'hidden', height: 8 }}>
          {Object.values(SEVERITY_LABELS).map((s, i) => (
            <div key={i} style={{ flex: 1, background: s.colour, opacity: 0.75 }} />
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          {Object.values(SEVERITY_LABELS).map((s, i) => (
            <span key={i} className="text-xs" style={{ color: s.colour, fontSize: '0.65rem' }}>{s.label}</span>
          ))}
        </div>
      </div>
    </div>
  )
}
