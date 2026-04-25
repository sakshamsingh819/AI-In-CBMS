/**
 * AntigravityProbabilityCard — Filled arc gauge (0–100%) showing the
 * softmax probability of the antigravity class.
 *
 * Threshold lines:
 *   30%  → WATCH   (amber)
 *   60%  → WARNING (orange)
 *   85%  → CONFIRM (deep purple glow)
 */

import React, { useMemo } from 'react'
import useStore from '../store/useStore'

const SIZE = 200
const CX = SIZE / 2
const CY = SIZE / 2
const RADIUS = 78
const STROKE_WIDTH = 14

// SVG arc helper: draws an arc for [startAngle, endAngle] in degrees
function describeArc(cx, cy, r, startAngle, endAngle) {
  const toRad = (deg) => (Math.PI / 180) * deg
  const x1 = cx + r * Math.cos(toRad(startAngle - 90))
  const y1 = cy + r * Math.sin(toRad(startAngle - 90))
  const x2 = cx + r * Math.cos(toRad(endAngle - 90))
  const y2 = cy + r * Math.sin(toRad(endAngle - 90))
  const largeArc = endAngle - startAngle > 180 ? 1 : 0
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`
}

function ThresholdTick({ angle, cx, cy, r, colour, label }) {
  const rad = ((angle - 90) * Math.PI) / 180
  const innerR = r - STROKE_WIDTH / 2 - 2
  const outerR = r + STROKE_WIDTH / 2 + 4
  const x1 = cx + innerR * Math.cos(rad)
  const y1 = cy + innerR * Math.sin(rad)
  const x2 = cx + outerR * Math.cos(rad)
  const y2 = cy + outerR * Math.sin(rad)
  const lx = cx + (outerR + 12) * Math.cos(rad)
  const ly = cy + (outerR + 12) * Math.sin(rad)
  return (
    <>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={colour} strokeWidth={2} />
      <text x={lx} y={ly} fill={colour} fontSize={9} textAnchor="middle" dominantBaseline="middle">
        {label}
      </text>
    </>
  )
}

function getGaugeColour(prob) {
  if (prob >= 0.85) return '#a855f7'
  if (prob >= 0.60) return '#f97316'
  if (prob >= 0.30) return '#f59e0b'
  return '#3b82f6'
}

function getLabel(prob) {
  if (prob >= 0.85) return { text: 'CONFIRMED', cls: 'badge-purple' }
  if (prob >= 0.60) return { text: 'WARNING',   cls: 'badge-amber' }
  if (prob >= 0.30) return { text: 'WATCH',     cls: 'badge-amber' }
  return { text: 'NORMAL', cls: 'badge-green' }
}

export default function AntigravityProbabilityCard() {
  const probability = useStore((s) => s.antigravityProbability)
  const antigravityEnabled = useStore((s) => s.antigravityEnabled)

  const prob = probability ?? 0

  // Arc spans from -135° to +135° (270° total sweep)
  const ARC_START = -135
  const ARC_END   = 135
  const fullSpan  = ARC_END - ARC_START
  const fillEnd   = ARC_START + fullSpan * prob

  const gaugeColour = getGaugeColour(prob)
  const label = getLabel(prob)

  // Threshold angles
  const thresholds = useMemo(() => [
    { pct: 0.30, colour: '#f59e0b', label: '30%' },
    { pct: 0.60, colour: '#f97316', label: '60%' },
    { pct: 0.85, colour: '#a855f7', label: '85%' },
  ], [])

  return (
    <div
      className={`glass-card p-6 fade-in ${prob >= 0.85 && antigravityEnabled ? 'glow-ag' : ''}`}
      style={{ textAlign: 'center' }}
    >
      <h2 className="font-semibold text-lg mb-4" style={{ color: 'var(--text-primary)' }}>
        ⚛ Antigravity Probability
      </h2>

      <svg width={SIZE} height={SIZE * 0.85} viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ display: 'block', margin: '0 auto', overflow: 'visible' }}>
        {/* Background arc track */}
        <path
          d={describeArc(CX, CY, RADIUS, ARC_START + 360, ARC_END + 360)}
          fill="none"
          stroke="rgba(148,163,184,0.1)"
          strokeWidth={STROKE_WIDTH}
          strokeLinecap="round"
        />

        {/* Filled arc */}
        {prob > 0.005 && (
          <path
            d={describeArc(CX, CY, RADIUS, ARC_START + 360, fillEnd + 360)}
            fill="none"
            stroke={gaugeColour}
            strokeWidth={STROKE_WIDTH}
            strokeLinecap="round"
            style={{
              filter: `drop-shadow(0 0 6px ${gaugeColour}88)`,
              transition: 'stroke 0.3s, d 0.5s',
            }}
          />
        )}

        {/* Threshold ticks */}
        {thresholds.map((t) => (
          <ThresholdTick
            key={t.pct}
            angle={ARC_START + fullSpan * t.pct + 360}
            cx={CX} cy={CY} r={RADIUS}
            colour={t.colour}
            label={t.label}
          />
        ))}

        {/* Centre text */}
        <text x={CX} y={CY - 8} fill="var(--text-primary)" fontSize={28} fontWeight="700"
          textAnchor="middle" dominantBaseline="middle" fontFamily="JetBrains Mono, monospace">
          {(prob * 100).toFixed(1)}%
        </text>
        <text x={CX} y={CY + 22} fill="var(--text-secondary)" fontSize={10}
          textAnchor="middle" dominantBaseline="middle">
          ANTIGRAVITY PROB.
        </text>
      </svg>

      <div style={{ marginTop: 8 }}>
        <span className={`badge ${label.cls}`}>{label.text}</span>
      </div>

      {!antigravityEnabled && (
        <p className="text-xs text-dimmed mt-4">
          Requires <code className="text-mono" style={{ color: 'var(--text-code)' }}>ANTIGRAVITY_ENABLED=true</code>
        </p>
      )}
    </div>
  )
}
