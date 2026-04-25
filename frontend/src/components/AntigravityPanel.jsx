/**
 * AntigravityPanel — Six horizontal gauge bars for all antigravity features.
 *
 * Each gauge shows: feature name, current value (monospace), reference band
 * (normal-gravity baseline ± 2σ), and a colour change when the value
 * exits the normal band.
 */

import React from 'react'
import useStore from '../store/useStore'

// Normal-gravity baselines: { mean, std, unit, description }
const BASELINES = {
  gli:  { mean: 0.98, std: 0.04, unit: '',    label: 'GLI',  desc: 'Gravity Load Index' },
  ser:  { mean: 0.06, std: 0.02, unit: '',    label: 'SER',  desc: 'Sub-sync Energy Ratio' },
  hcc:  { mean: 0.05, std: 0.03, unit: '',    label: 'HCC',  desc: 'Harmonic Cancellation Coeff.' },
  cpc:  { mean: 0.75, std: 0.08, unit: '',    label: 'CPC',  desc: 'Cross-axis Phase Coherence' },
  blza: { mean: 1.50, std: 0.25, unit: '',    label: 'BLZA', desc: 'Bearing Load Zone Asymmetry' },
  gpi:  { mean: 0.98, std: 0.12, unit: '×', label: 'GPI',  desc: 'Gyroscopic Precession Index' },
}

// Percentile clamp for rendering a [0–1] fill ratio
function normalisedFill(key, value) {
  const ranges = {
    gli:  [0, 2.0],
    ser:  [0, 0.6],
    hcc:  [-0.5, 1.0],
    cpc:  [0, 1.0],
    blza: [0, 5.0],
    gpi:  [0, 2.0],
  }
  const [lo, hi] = ranges[key] || [0, 1]
  return Math.max(0, Math.min(1, (value - lo) / (hi - lo)))
}

function GaugeRow({ featureKey, value }) {
  const { mean, std, label, desc, unit } = BASELINES[featureKey]
  const lo2σ = mean - 2 * std
  const hi2σ = mean + 2 * std
  const isAnomalous = value < lo2σ || value > hi2σ

  const fillRatio = normalisedFill(featureKey, value)
  const fillColour = isAnomalous ? '#a855f7' : '#3b82f6'
  const glowColour = isAnomalous ? 'rgba(168,85,247,0.4)' : 'rgba(59,130,246,0.2)'

  // Band reference fill ratios
  const bandLo = normalisedFill(featureKey, lo2σ)
  const bandHi = normalisedFill(featureKey, hi2σ)

  return (
    <div className="gauge-row" style={{ marginBottom: 14 }}>
      <div className="flex justify-between items-center mb-2">
        <div>
          <span
            className="font-semibold text-sm"
            style={{ color: isAnomalous ? '#a855f7' : 'var(--text-primary)', letterSpacing: '0.05em' }}
          >
            {label}
          </span>
          <span className="text-xs text-dimmed" style={{ marginLeft: 8 }}>{desc}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            className="text-mono font-medium"
            style={{
              fontSize: '0.875rem',
              color: isAnomalous ? '#a855f7' : 'var(--text-primary)',
              minWidth: 52,
              textAlign: 'right',
            }}
          >
            {value.toFixed(3)}{unit}
          </span>
          {isAnomalous && (
            <span className="badge badge-purple" style={{ fontSize: '0.65rem', padding: '2px 7px' }}>
              ⚠ ANOMALY
            </span>
          )}
        </div>
      </div>

      {/* Track with reference band */}
      <div style={{ position: 'relative', height: 10, borderRadius: 100, background: 'rgba(148,163,184,0.08)', overflow: 'visible' }}>
        {/* Normal-gravity reference band (±2σ) */}
        <div
          style={{
            position: 'absolute',
            left: `${bandLo * 100}%`,
            width: `${(bandHi - bandLo) * 100}%`,
            height: '100%',
            background: 'rgba(148,163,184,0.18)',
            borderRadius: 100,
          }}
        />
        {/* Actual value fill */}
        <div
          style={{
            position: 'absolute',
            left: 0,
            width: `${fillRatio * 100}%`,
            height: '100%',
            background: `linear-gradient(90deg, ${fillColour}cc, ${fillColour})`,
            borderRadius: 100,
            boxShadow: isAnomalous ? `0 0 10px ${glowColour}` : 'none',
            transition: 'width 0.4s ease, background 0.3s ease',
          }}
        />
      </div>

      <div className="flex justify-between" style={{ marginTop: 3 }}>
        <span className="text-xs text-dimmed">Baseline: {mean.toFixed(2)} ± {(2 * std).toFixed(2)}</span>
        <span className="text-xs text-dimmed">
          {lo2σ.toFixed(2)} – {hi2σ.toFixed(2)}
        </span>
      </div>
    </div>
  )
}

export default function AntigravityPanel() {
  const antigravityFeatures = useStore((s) => s.antigravityFeatures)
  const antigravityEnabled = useStore((s) => s.antigravityEnabled)

  if (!antigravityEnabled) {
    return (
      <div className="glass-card p-6 fade-in" style={{ textAlign: 'center' }}>
        <span style={{ fontSize: '2rem' }}>🔬</span>
        <p className="text-muted mt-4" style={{ fontSize: '0.875rem' }}>
          Antigravity detection is disabled.
        </p>
        <p className="text-dimmed" style={{ fontSize: '0.75rem', marginTop: 4 }}>
          Set <code className="text-mono" style={{ color: 'var(--text-code)' }}>ANTIGRAVITY_ENABLED=true</code> in <code>.env</code>
        </p>
      </div>
    )
  }

  const features = antigravityFeatures || { gli: 0.98, ser: 0.06, hcc: 0.05, cpc: 0.75, blza: 1.50, gpi: 0.98 }

  return (
    <div className="glass-card p-6 fade-in">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="font-semibold text-lg" style={{ color: 'var(--text-primary)' }}>
            ⚛ Antigravity Feature Gauges
          </h2>
          <p className="text-xs text-dimmed mt-2">
            Reference band = normal-gravity baseline ± 2σ &nbsp;|&nbsp; Purple = anomalous
          </p>
        </div>
        {!antigravityFeatures && (
          <span className="badge badge-blue">Demo values</span>
        )}
      </div>

      {/* 6 gauge rows */}
      {Object.keys(BASELINES).map((key) => (
        <GaugeRow key={key} featureKey={key} value={features[key]} />
      ))}
    </div>
  )
}
