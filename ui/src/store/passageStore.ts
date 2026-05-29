import { create } from 'zustand'
import { analyzePassage as apiAnalyzePassage, getLenses } from '../api/client'

export interface TokenPoint {
  position: number
  token: string
  entropy: number
  superpositionScore: number
  activeFields: Record<string, number>
  topField: string
}

export interface AnamnesisResult {
  retrievedLenses: [string, number][]
  retrievedNodes: Array<{
    surface: string
    type: string
    meaning?: string
    score: number
  }>
  novelConnections: Array<{
    from: string
    to: string
    path: string[]
    noveltyScore: number
  }>
  narrative: string
}

export interface ContrastiveResult {
  passage: string
  page: number
  line: number
  tokensByLens: Record<string, TokenPoint[]>
  divergenceMap: number[][]
  agreementMatrix: number[][]
  superpositionPositions: number[]
  lensNames: string[]
  anamnesis?: AnamnesisResult
}

export interface PassageStore {
  result: ContrastiveResult | null
  isLoading: boolean
  error: string | null
  selectedPosition: number | null
  selectedLenses: string[]
  availableLenses: string[]

  setResult: (r: ContrastiveResult) => void
  setLoading: (v: boolean) => void
  setError: (e: string | null) => void
  setSelectedPosition: (p: number | null) => void
  setSelectedLenses: (l: string[]) => void
  loadAvailableLenses: () => Promise<void>

  analyzePassage: (
    passage: string,
    page: number,
    line: number,
    lenses: string[],
  ) => Promise<void>
}

const DEFAULT_LENSES = [
  'mythological',
  'kabbalistic',
  'psychoanalytic',
  'historical',
  'linguistic',
]

export const usePassageStore = create<PassageStore>((set, get) => ({
  result: null,
  isLoading: false,
  error: null,
  selectedPosition: null,
  selectedLenses: DEFAULT_LENSES,
  availableLenses: DEFAULT_LENSES,

  setResult: (r) => set({ result: r }),
  setLoading: (v) => set({ isLoading: v }),
  setError: (e) => set({ error: e }),
  setSelectedPosition: (p) => set({ selectedPosition: p }),
  setSelectedLenses: (l) => set({ selectedLenses: l }),

  loadAvailableLenses: async () => {
    try {
      const lenses = await getLenses()
      if (lenses.length > 0) {
        set({ availableLenses: lenses, selectedLenses: lenses })
      }
    } catch {
      // Keep defaults on error
    }
  },

  analyzePassage: async (passage, page, line, lenses) => {
    const { setLoading, setError, setResult } = get()
    setLoading(true)
    setError(null)
    try {
      const result = await apiAnalyzePassage({ passage, page, line, lenses })
      setResult(result)
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Analysis failed. Please retry.'
      setError(message)
    } finally {
      setLoading(false)
    }
  },
}))
