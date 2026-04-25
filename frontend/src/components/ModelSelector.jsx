/**
 * ModelSelector — Dropdown to select the active inference model.
 *
 * Shows for each model:
 *   - Name, version, load status
 *   - Number of fault classes
 *   - antigravity_f1 (when ANTIGRAVITY_ENABLED=true)
 *   - Tooltip with full capability details
 */

import React, { useEffect, useState } from 'react'
import useStore from '../store/useStore'
import { api } from '../api/client'

export default function ModelSelector() {
  const models = useStore((s) => s.models)
  const setModels = useStore((s) => s.setModels)
  const activeModel = useStore((s) => s.activeModel)
  const setActiveModel = useStore((s) => s.setActiveModel)
  const antigravityEnabled = useStore((s) => s.antigravityEnabled)
  const [tooltip, setTooltip] = useState(null)

  useEffect(() => {
    api.models().then((data) => setModels(data.models)).catch(() => {})
  }, [])

  return (
    <div className="glass-card p-4">
      <p className="text-xs text-dimmed font-medium mb-4" style={{ textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        Inference Model
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {models.length === 0 && (
          <p className="text-xs text-muted">Loading models…</p>
        )}
        {models.map((model) => (
          <button
            key={model.name}
            id={`model-btn-${model.name.toLowerCase().replace(/\s+/g, '-')}`}
            onClick={() => setActiveModel(model.name)}
            onMouseEnter={() => setTooltip(model.name)}
            onMouseLeave={() => setTooltip(null)}
            style={{
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '10px 14px',
              borderRadius: 10,
              border: activeModel === model.name
                ? '1.5px solid var(--accent-blue)'
                : '1.5px solid var(--border-subtle)',
              background: activeModel === model.name
                ? 'rgba(59,130,246,0.08)'
                : 'transparent',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              textAlign: 'left',
              width: '100%',
            }}
          >
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>
                  {model.name}
                </span>
                <span className="text-xs text-dimmed">v{model.version || '1'}</span>
                {!model.loaded && (
                  <span className="badge badge-amber" style={{ fontSize: '0.6rem', padding: '1px 6px' }}>stub</span>
                )}
              </div>
              <div className="flex items-center gap-3 mt-2">
                <span className="text-xs text-dimmed">{model.feature_vector_length} features</span>
                <span className="text-xs text-dimmed">{model.fault_classes?.length} classes</span>
                {antigravityEnabled && model.antigravity_capable && (
                  <span className="badge badge-purple" style={{ fontSize: '0.6rem', padding: '1px 6px' }}>
                    AG-capable
                  </span>
                )}
              </div>
            </div>

            {/* Antigravity F1 score */}
            {antigravityEnabled && model.antigravity_f1 != null && (
              <div style={{ textAlign: 'right' }}>
                <p className="text-xs text-dimmed">AG F1</p>
                <p className="text-mono font-semibold" style={{ color: '#a855f7', fontSize: '0.9rem' }}>
                  {model.antigravity_f1.toFixed(2)}
                </p>
              </div>
            )}

            {/* Tooltip */}
            {tooltip === model.name && (
              <div style={{
                position: 'absolute',
                bottom: 'calc(100% + 8px)',
                left: 0,
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border-dim)',
                borderRadius: 10,
                padding: '12px 16px',
                zIndex: 100,
                minWidth: 260,
                boxShadow: 'var(--shadow-md)',
                pointerEvents: 'none',
              }}>
                <p className="font-semibold text-sm mb-3" style={{ color: 'var(--text-primary)' }}>
                  {model.name} — Capabilities
                </p>
                <p className="text-xs text-muted mb-2">
                  Classes: {model.fault_classes?.join(', ')}
                </p>
                <p className="text-xs text-muted mb-2">
                  Feature vector: {model.feature_vector_length} dimensions
                </p>
                {model.antigravity_capable && (
                  <p className="text-xs" style={{ color: '#a855f7' }}>
                    ⚛ Antigravity-capable (7-class)
                  </p>
                )}
                {model.antigravity_f1 != null && (
                  <p className="text-xs mt-2" style={{ color: '#a855f7' }}>
                    Antigravity F1 (synthetic): {model.antigravity_f1.toFixed(3)}
                  </p>
                )}
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
