import React from 'react'
import { usePassageStore } from './store/passageStore'
import { PassageInput } from './components/PassageInput/PassageInput'
import { EntropyWaveform } from './components/EntropyWaveform/EntropyWaveform'
import { LensComparison } from './components/LensComparison/LensComparison'
import { SuperpositionMap } from './components/SuperpositionMap/SuperpositionMap'
import { GraphExplorer } from './components/GraphExplorer/GraphExplorer'
import { AnamnesisPanel } from './components/AnamnesisPanel/AnamnesisPanel'
import { ErrorBanner } from './components/common/ErrorBanner'
import { LoadingSpinner } from './components/common/LoadingSpinner'

interface PanelProps {
  title: string
  badge?: string
  children: React.ReactNode
  className?: string
}

const Panel: React.FC<PanelProps> = ({ title, badge, children, className = '' }) => (
  <div className={`panel ${className}`}>
    <div className="panel-header">
      <span className="panel-title">{title}</span>
      {badge && (
        <span className="text-xs font-mono text-gray-600 bg-gray-900/50 px-2 py-0.5 rounded border border-gray-700/50">
          {badge}
        </span>
      )}
    </div>
    <div className="panel-body">{children}</div>
  </div>
)

const GlobalLoadingOverlay: React.FC = () => {
  const isLoading = usePassageStore((s) => s.isLoading)
  if (!isLoading) return null
  return (
    <div
      className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm
                 flex flex-col items-center justify-center gap-4 pointer-events-none"
    >
      <LoadingSpinner size="lg" />
      <span className="font-mono text-sm text-sky-400 animate-pulse tracking-wider">
        Running contrastive lens analysis…
      </span>
    </div>
  )
}

const App: React.FC = () => {
  const { error, result, isLoading } = usePassageStore()

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      <GlobalLoadingOverlay />

      {/* ── Top bar ────────────────────────────────────────────────────────── */}
      <header className="flex-shrink-0 border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm">
        <div className="max-w-screen-2xl mx-auto px-5 py-3 flex items-start gap-6">
          {/* Logo */}
          <div className="flex-shrink-0 flex flex-col justify-center pt-0.5">
            <div className="flex items-baseline gap-2">
              <span
                className="font-mono font-bold text-2xl tracking-[0.25em] text-sky-400
                           drop-shadow-[0_0_8px_rgba(56,189,248,0.4)]"
              >
                WAKE
              </span>
              <span className="text-gray-600 font-mono text-xs">v0.1</span>
            </div>
            <span className="text-gray-600 font-mono text-[9px] tracking-widest uppercase">
              interpretability interface
            </span>
          </div>

          {/* Divider */}
          <div className="w-px self-stretch bg-gray-800 flex-shrink-0" />

          {/* Passage input */}
          <div className="flex-1 min-w-0">
            <PassageInput />
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="max-w-screen-2xl mx-auto px-5 pb-3">
            <ErrorBanner message={error} />
          </div>
        )}
      </header>

      {/* ── Main content ────────────────────────────────────────────────────── */}
      <main className="flex-1 min-h-0 flex flex-col">
        {/* Status bar */}
        {result && !isLoading && (
          <div
            className="flex-shrink-0 px-5 py-1.5 bg-gray-900/50 border-b border-gray-800/60
                       flex items-center gap-4 text-xs font-mono text-gray-500"
          >
            <span className="text-green-500">◆</span>
            <span>
              &ldquo;{result.passage.slice(0, 60)}
              {result.passage.length > 60 ? '…' : ''}&rdquo;
            </span>
            <span className="text-gray-700">·</span>
            <span>
              pg.{result.page} ln.{result.line}
            </span>
            <span className="text-gray-700">·</span>
            <span>{result.lensNames.length} lenses</span>
            <span className="text-gray-700">·</span>
            <span>
              {Object.values(result.tokensByLens)[0]?.length ?? 0} tokens
            </span>
            {result.superpositionPositions.length > 0 && (
              <>
                <span className="text-gray-700">·</span>
                <span className="text-yellow-500">
                  ◈ {result.superpositionPositions.length} superposition
                </span>
              </>
            )}
          </div>
        )}

        {/* 2×2 panel grid */}
        <div className="flex-1 min-h-0 grid grid-cols-2 grid-rows-2 gap-3 p-3">
          <Panel
            title="Entropy Waveform"
            badge="per-token · all lenses"
            className="min-h-0"
          >
            <EntropyWaveform />
          </Panel>

          <Panel
            title="Lens Divergence"
            badge="heatmap · click cell for detail"
            className="min-h-0"
          >
            <LensComparison />
          </Panel>

          <Panel
            title="Residual Stream Geometry"
            badge="PCA · superposition tokens highlighted"
            className="min-h-0"
          >
            <SuperpositionMap />
          </Panel>

          <Panel
            title="Graph Explorer"
            badge="force-directed · click to expand"
            className="min-h-0"
          >
            <GraphExplorer />
          </Panel>
        </div>
      </main>

      {/* ── Bottom drawer: Anamnesis ─────────────────────────────────────── */}
      <AnamnesisPanel />
    </div>
  )
}

export default App
