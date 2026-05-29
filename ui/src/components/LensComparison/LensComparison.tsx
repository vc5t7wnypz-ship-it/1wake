import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import type { PlotMouseEvent } from 'plotly.js'
import { usePassageStore } from '../../store/passageStore'

interface DetailModalProps {
  lensA: string
  lensB: string
  tokenStr: string
  position: number
  onClose: () => void
}

const DetailModal: React.FC<DetailModalProps> = ({
  lensA,
  lensB,
  tokenStr,
  position,
  onClose,
}) => {
  const { result } = usePassageStore()
  const tokensA = result?.tokensByLens[lensA] ?? []
  const tokensB = result?.tokensByLens[lensB] ?? []
  const tA = tokensA.find((t) => t.position === position)
  const tB = tokensB.find((t) => t.position === position)

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-md w-full mx-4
                   shadow-2xl animate-fade-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-mono text-sm font-semibold text-gray-200">
            Token:{' '}
            <span className="text-sky-400">
              &ldquo;{tokenStr}&rdquo;
            </span>{' '}
            @ pos {position}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 font-bold text-lg leading-none"
          >
            ×
          </button>
        </div>
        <div className="grid grid-cols-2 gap-4">
          {[
            { name: lensA, tok: tA },
            { name: lensB, tok: tB },
          ].map(({ name, tok }) => (
            <div key={name} className="bg-gray-800 rounded-lg p-3">
              <div className="text-xs font-mono text-gray-400 mb-2 font-semibold uppercase tracking-wider">
                {name}
              </div>
              {tok ? (
                <>
                  <div className="text-xs font-mono text-gray-300 mb-1">
                    entropy:{' '}
                    <span className="text-sky-400">
                      {tok.entropy.toFixed(3)}
                    </span>
                  </div>
                  <div className="text-xs font-mono text-gray-300 mb-1">
                    superposition:{' '}
                    <span className="text-yellow-400">
                      {tok.superpositionScore.toFixed(3)}
                    </span>
                  </div>
                  <div className="text-xs font-mono text-gray-300">
                    top field:{' '}
                    <span className="text-purple-400">{tok.topField}</span>
                  </div>
                  <div className="mt-2 text-xs text-gray-500">
                    Active fields:
                  </div>
                  <div className="mt-1 flex flex-col gap-0.5">
                    {Object.entries(tok.activeFields)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 4)
                      .map(([field, val]) => (
                        <div
                          key={field}
                          className="flex items-center gap-2 text-xs font-mono"
                        >
                          <div className="flex-1 text-gray-400 truncate">
                            {field}
                          </div>
                          <div
                            className="h-1.5 rounded-full bg-sky-600"
                            style={{ width: `${Math.round(val * 60)}px` }}
                          />
                          <div className="text-gray-500 w-8 text-right">
                            {val.toFixed(2)}
                          </div>
                        </div>
                      ))}
                  </div>
                </>
              ) : (
                <div className="text-xs text-gray-600">No data</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export const LensComparison: React.FC = () => {
  const { result } = usePassageStore()
  const [modal, setModal] = useState<{
    lensRow: number
    tokenCol: number
  } | null>(null)

  if (!result) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <span className="text-gray-600 font-mono text-xs">
          Submit a passage to view lens divergence
        </span>
      </div>
    )
  }

  const { lensNames, divergenceMap, tokensByLens } = result
  const firstLensTokens = tokensByLens[lensNames[0]] ?? []
  const xLabels = firstLensTokens.map((t) => t.token.slice(0, 8))

  // divergenceMap: rows = lenses, cols = positions
  const zData: number[][] =
    divergenceMap.length > 0
      ? divergenceMap
      : lensNames.map((lens) =>
          (tokensByLens[lens] ?? []).map((t) => t.entropy / 12),
        )

  const handleClick = (e: Readonly<PlotMouseEvent>) => {
    const pt = e.points[0]
    if (!pt) return
    // Plotly heatmap pointIndex is [row, col] but typed as number in some versions
    const idx = pt.pointIndex as unknown as number[]
    setModal({
      lensRow: Array.isArray(idx) ? (idx[0] ?? 0) : 0,
      tokenCol: Array.isArray(idx) ? (idx[1] ?? 0) : 0,
    })
  }

  const closeModal = () => setModal(null)

  const modalData =
    modal !== null
      ? {
          lensA: lensNames[modal.lensRow] ?? lensNames[0],
          lensB:
            lensNames[(modal.lensRow + 1) % lensNames.length] ?? lensNames[0],
          tokenStr: xLabels[modal.tokenCol] ?? '',
          position: firstLensTokens[modal.tokenCol]?.position ?? 0,
        }
      : null

  return (
    <div className="w-full h-full flex flex-col">
      <Plot
        data={[
          {
            type: 'heatmap',
            z: zData,
            x: xLabels,
            y: lensNames,
            colorscale: 'RdBu_r' as unknown as Plotly.ColorScale,
            zmin: 0,
            zmax: 1,
            hoverongaps: false,
            hovertemplate:
              'lens: %{y}<br>token: %{x}<br>divergence: %{z:.3f}<extra></extra>',
            colorbar: {
              thickness: 10,
              len: 0.8,
              tickfont: { color: '#9ca3af', family: 'JetBrains Mono', size: 9 },
              title: {
                text: 'div',
                font: { color: '#9ca3af', family: 'JetBrains Mono', size: 9 },
              },
            },
          } as Plotly.Data,
        ]}
        layout={
          {
            autosize: true,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { l: 90, r: 60, t: 8, b: 60 },
            font: { family: 'JetBrains Mono, monospace', color: '#9ca3af', size: 10 },
            xaxis: {
              tickangle: -45,
              tickfont: { size: 9, color: '#6b7280' },
              gridcolor: '#1f2937',
              tickcolor: '#374151',
              linecolor: '#374151',
            },
            yaxis: {
              tickfont: { size: 9, color: '#6b7280' },
              gridcolor: '#1f2937',
              tickcolor: '#374151',
              linecolor: '#374151',
            },
          } as Partial<Plotly.Layout>
        }
        config={{ responsive: true, displayModeBar: false }}
        onClick={handleClick}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
      {modalData && (
        <DetailModal
          lensA={modalData.lensA}
          lensB={modalData.lensB}
          tokenStr={modalData.tokenStr}
          position={modalData.position}
          onClose={closeModal}
        />
      )}
    </div>
  )
}
