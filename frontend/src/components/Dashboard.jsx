/**
 * Dashboard — Main layout component for the Condition Monitoring System.
 *
 * Layout:
 *   Header: logo + connection status + AG toggle
 *   Left sidebar: ModelSelector + SeverityGauge
 *   Main area: FaultCard + AntigravityProbabilityCard
 *   Bottom: AntigravityPanel (full width)
 *   Demo controls: run synthetic analysis from the UI
 */

import React, { useState, useCallback } from 'react'
import useStore from '../store/useStore'
import { api } from '../api/client'
import FaultCard from './FaultCard'
import AntigravityPanel from './AntigravityPanel'
import AntigravityProbabilityCard from './AntigravityProbabilityCard'
import ModelSelector from './ModelSelector'
import SeverityGauge from './SeverityGauge'

// ── Synthetic demo signal generator ───────────────────────────────────────────
function makeDemoSignal(type = 'normal') {
  const N = 1024
  const FS = 26667
  const FR = 25   // 1500 RPM
  const G  = 9.81
  const arr = (fn) => Array.from({ length: N }, (_, i) => fn(i / FS))

  if (type === 'antigravity') {
    const alpha = 0.05
    return {
      signal_x: arr((t) => (0.1 + 0.9 * alpha) * Math.sin(2 * Math.PI * FR * t)
        + 0.12 * Math.sin(2 * Math.PI * 0.25 * FR * t)
        + 0.15 * Math.sin(2 * Math.PI * 0.43 * FR * t)
        + (Math.random() - 0.5) * 0.05),
      signal_y: arr((t) => (0.1 + 0.9 * alpha) * Math.sin(2 * Math.PI * FR * t + Math.PI / 4)
        + (Math.random() - 0.5) * 0.05),
      signal_z: arr((t) => G * alpha + 0.1 * Math.sin(2 * Math.PI * FR * t)
        + (Math.random() - 0.5) * 0.05),
    }
  }
  return {
    signal_x: arr((t) => Math.sin(2 * Math.PI * FR * t) + 0.3 * Math.sin(4 * Math.PI * FR * t) + (Math.random() - 0.5) * 0.05),
    signal_y: arr((t) => Math.sin(2 * Math.PI * FR * t + Math.PI / 2) + (Math.random() - 0.5) * 0.05),
    signal_z: arr((t) => G + 0.3 * Math.sin(2 * Math.PI * FR * t) + (Math.random() - 0.5) * 0.05),
  }
}

export default function Dashboard() {
  const sessionId        = useStore((s) => s.sessionId)
  const windowIndex      = useStore((s) => s.windowIndex)
  const shaftRpm         = useStore((s) => s.shaftRpm)
  const antigravityEnabled = useStore((s) => s.antigravityEnabled)
  const isAnalysing      = useStore((s) => s.isAnalysing)
  const analysisError    = useStore((s) => s.analysisError)

  const setClassification  = useStore((s) => s.setClassification)
  const setAntigravityResult = useStore((s) => s.setAntigravityResult)
  const clearAntigravity   = useStore((s) => s.clearAntigravityResult)
  const setAnalysing       = useStore((s) => s.setAnalysing)
  const setAnalysisError   = useStore((s) => s.setAnalysisError)
  const setAntigravityEnabled = useStore((s) => s.setAntigravityEnabled)
  const incrementWindow    = useStore((s) => s.incrementWindow)

  const [demoMode, setDemoMode] = useState('normal')

  const runAnalysis = useCallback(async () => {
    setAnalysing(true)
    setAnalysisError(null)
    try {
      const signal = makeDemoSignal(demoMode)
      const base = { session_id: sessionId, window_index: windowIndex, shaft_rpm: shaftRpm, ...signal }

      // Run classify
      const clf = await api.classify(base)
      setClassification(clf)

      // Run antigravity analysis if enabled
      if (antigravityEnabled) {
        const ag = await api.antigravity(base)
        setAntigravityResult(ag)
      } else {
        clearAntigravity()
      }

      incrementWindow()
    } catch (err) {
      setAnalysisError(err.message)
    } finally {
      setAnalysing(false)
    }
  }, [sessionId, windowIndex, shaftRpm, antigravityEnabled, demoMode])

  return (
    <div style={{ minHeight: '100vh', padding: '24px', maxWidth: 1400, margin: '0 auto' }}>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 28 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{
              width: 44, height: 44, borderRadius: 12,
              background: 'linear-gradient(135deg, #3b82f6, #4B0082)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '1.4rem', boxShadow: 'var(--shadow-glow-blue)',
            }}>
              ⚙️
            </div>
            <div>
              <h1 className="text-xl font-bold" style={{ color: 'var(--text-primary)', lineHeight: 1.2 }}>
                Condition Monitoring System
              </h1>
              <p className="text-xs text-dimmed mt-1">
                STWIN.box MEMS · cGAN-Hybrid Classifier · v1.1.0-antigravity
              </p>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          {/* RPM control */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <label className="text-xs text-dimmed">RPM:</label>
            <input
              type="number"
              value={shaftRpm}
              onChange={(e) => useStore.getState().setShaftRpm(Number(e.target.value))}
              style={{
                width: 75, padding: '5px 8px', borderRadius: 8,
                background: 'var(--bg-elevated)', border: '1px solid var(--border-dim)',
                color: 'var(--text-primary)', fontSize: '0.85rem', fontFamily: 'var(--font-mono)',
              }}
            />
          </div>

          {/* Demo mode */}
          <select
            value={demoMode}
            onChange={(e) => setDemoMode(e.target.value)}
            style={{
              padding: '5px 10px', borderRadius: 8,
              background: 'var(--bg-elevated)', border: '1px solid var(--border-dim)',
              color: 'var(--text-primary)', fontSize: '0.85rem', cursor: 'pointer',
            }}
          >
            <option value="normal">Demo: Normal Gravity</option>
            <option value="antigravity">Demo: Antigravity</option>
          </select>

          {/* Antigravity toggle */}
          <button
            id="antigravity-toggle"
            onClick={() => setAntigravityEnabled(!antigravityEnabled)}
            style={{
              padding: '6px 14px', borderRadius: 10, cursor: 'pointer',
              background: antigravityEnabled ? 'rgba(75,0,130,0.25)' : 'var(--bg-elevated)',
              border: antigravityEnabled ? '1.5px solid rgba(75,0,130,0.6)' : '1.5px solid var(--border-dim)',
              color: antigravityEnabled ? '#c084fc' : 'var(--text-secondary)',
              fontSize: '0.8rem', fontWeight: 600,
              transition: 'all 0.25s ease',
              boxShadow: antigravityEnabled ? 'var(--shadow-glow-purple)' : 'none',
            }}
          >
            ⚛ Antigravity {antigravityEnabled ? 'ON' : 'OFF'}
          </button>

          {/* Analyse button */}
          <button
            id="run-analysis-btn"
            onClick={runAnalysis}
            disabled={isAnalysing}
            style={{
              padding: '8px 20px', borderRadius: 10, cursor: isAnalysing ? 'not-allowed' : 'pointer',
              background: isAnalysing
                ? 'var(--bg-elevated)'
                : 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
              border: 'none', color: '#fff', fontSize: '0.875rem', fontWeight: 600,
              opacity: isAnalysing ? 0.6 : 1,
              boxShadow: isAnalysing ? 'none' : 'var(--shadow-glow-blue)',
              transition: 'all 0.2s ease',
            }}
          >
            {isAnalysing ? '⏳ Analysing…' : '▶ Run Analysis'}
          </button>
        </div>
      </header>

      {/* Error banner */}
      {analysisError && (
        <div style={{
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: 10, padding: '10px 16px', marginBottom: 20,
          color: 'var(--accent-red)', fontSize: '0.85rem',
        }}>
          ⚠ {analysisError}
        </div>
      )}

      {/* ── Main grid ──────────────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 20, alignItems: 'start' }}>

        {/* Left sidebar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <ModelSelector />
          <SeverityGauge />
        </div>

        {/* Main content */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Top row: FaultCard + Probability card */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20, alignItems: 'start' }}>
            <FaultCard />
            <AntigravityProbabilityCard />
          </div>

          {/* Bottom: full-width antigravity panel */}
          <AntigravityPanel />
        </div>
      </div>

      {/* Footer */}
      <footer style={{ marginTop: 32, textAlign: 'center' }}>
        <p className="text-xs text-dimmed">
          CMS v1.1.0-antigravity · Session: <code className="text-mono" style={{ color: 'var(--text-code)' }}>{sessionId}</code>
          &nbsp;·&nbsp;Window #{windowIndex}
        </p>
      </footer>
    </div>
  )
}
