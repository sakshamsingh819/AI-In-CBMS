/**
 * Zustand global store for the Condition Monitoring System dashboard.
 * State is organized into three slices: session, analysis, antigravity.
 */

import { create } from 'zustand'

const useStore = create((set, get) => ({
  // ── Session state ──────────────────────────────────────────────────────────
  sessionId: `sess_${Date.now()}`,
  windowIndex: 0,
  shaftRpm: 1500,
  activeModel: 'Random Forest',
  antigravityEnabled: false,

  setShaftRpm: (rpm) => set({ shaftRpm: rpm }),
  setActiveModel: (model) => set({ activeModel: model }),
  setAntigravityEnabled: (enabled) => set({ antigravityEnabled: enabled }),
  incrementWindow: () => set((s) => ({ windowIndex: s.windowIndex + 1 })),

  // ── Classification result ──────────────────────────────────────────────────
  classification: null,
  /**
   * @type {{
   *   session_id: string,
   *   window_index: number,
   *   predicted_class: string,
   *   confidence: number,
   *   class_probabilities: Record<string, number>,
   *   model_used: string,
   *   feature_vector_length: number,
   *   antigravity_probability: number | null
   * } | null}
   */
  setClassification: (result) => set({ classification: result }),

  // ── Feature vector ─────────────────────────────────────────────────────────
  features: null,
  featureNames: [],
  setFeatures: (features, names) => set({ features, featureNames: names }),

  // ── Antigravity state ──────────────────────────────────────────────────────
  /**
   * @type {{ gli: number, ser: number, hcc: number, cpc: number, blza: number, gpi: number } | null}
   */
  antigravityFeatures: null,
  antigravityProbability: null,   // number | null
  antigravityRegime: null,        // 'normal_gravity' | 'reduced_gravity' | 'near_zero_gravity' | null

  setAntigravityResult: (result) =>
    set({
      antigravityFeatures: {
        gli:  result.gli,
        ser:  result.ser,
        hcc:  result.hcc,
        cpc:  result.cpc,
        blza: result.blza,
        gpi:  result.gpi,
      },
      antigravityProbability: result.antigravity_probability,
      antigravityRegime: result.regime,
    }),

  clearAntigravityResult: () =>
    set({
      antigravityFeatures: null,
      antigravityProbability: null,
      antigravityRegime: null,
    }),

  // ── Models metadata ────────────────────────────────────────────────────────
  models: [],
  setModels: (models) => set({ models }),

  // ── UI state ───────────────────────────────────────────────────────────────
  isAnalysing: false,
  analysisError: null,
  setAnalysing: (v) => set({ isAnalysing: v }),
  setAnalysisError: (err) => set({ analysisError: err }),
}))

export default useStore
