import React from 'react'
import Plot from 'react-plotly.js'
import { usePassageStore } from '../../store/passageStore'

const PLOTLY_SYMBOLS = [
  'circle',
  'square',
  'diamond',
  'cross',
  'x',
  'triangle-up',
  'triangle-down',
  'pentagon',
  'star',
  'hexagon',
] as const

// Simple deterministic 2-D projection from entropy + superpositionScore.
// In production, PCA coords come from the API's divergenceMap rows.
function projectToPCA(
  entropies: number[],
  superScores: number[],
  divergenceRow: number[],
): [number, number][] {
  return entropies.map((e, i) => {
    const x = divergenceRow[i] !== undefined ? divergenceRow[i] * 2 - 1 : e * 0.1 - 0.6
    const y = superScores[i] * 2 - 1
    return [x, y]
  })
}

export const SuperpositionMap: React.FC = () => {
  const { result } = usePassageStore()

  if (!result) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <span className="text-gray-600 font-mono text-xs">
          Submit a passage to view residual stream geometry
        </span>
      </div>
    )
  }

  const { lensNames, tokensByLens, divergenceMap, superpositionPositions } =
    result

  const traces: Plotly.Data[] = lensNames.map((lens, lensIdx) => {
    const tokens = tokensByLens[lens] ?? []
    const divRow =
      divergenceMap[lensIdx] ?? tokens.map((t) => t.entropy / 12)

    const coords = projectToPCA(
      tokens.map((t) => t.entropy),
      tokens.map((t) => t.superpositionScore),
      divRow,
    )

    const isSuperList = tokens.map((t) =>
      superpositionPositions.includes(t.position),
    )

    const tokenSymbol = PLOTLY_SYMBOLS[lensIdx % PLOTLY_SYMBOLS.length]

    return {
      type: 'scatter',
      mode: 'markers+text' as Plotly.PlotData['mode'],
      name: lens,
      x: coords.map(([x]) => x),
      y: coords.map(([, y]) => y),
      text: tokens.map((t, i) =>
        isSuperList[i] ? t.token.slice(0, 10) : '',
      ),
      textposition: 'top center' as Plotly.PlotData['textposition'],
      textfont: { size: 8, color: '#f59e0b', family: 'JetBrains Mono' },
      marker: {
        symbol: tokenSymbol as Plotly.PlotData['marker'],
        size: isSuperList.map((s) => (s ? 14 : 8)),
        color: isSuperList.map((s) =>
          s ? '#f59e0b' : '#1e3a5f',
        ),
        opacity: isSuperList.map((s) => (s ? 1.0 : 0.65)),
        line: {
          color: isSuperList.map((s) => (s ? '#fbbf24' : 'transparent')),
          width: isSuperList.map((s) => (s ? 1.5 : 0)),
        },
      },
      customdata: tokens.map((t, i) => [
        t.token,
        lens,
        t.superpositionScore.toFixed(3),
        isSuperList[i] ? '★ superposition' : '',
      ]),
      hovertemplate:
        '<b>%{customdata[0]}</b><br>' +
        'lens: %{customdata[1]}<br>' +
        'superposition: %{customdata[2]}<br>' +
        '%{customdata[3]}' +
        '<extra></extra>',
    } as unknown as Plotly.Data
  })

  return (
    <div className="w-full h-full">
      <Plot
        data={traces}
        layout={
          {
            autosize: true,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'rgba(17,24,39,0.6)',
            margin: { l: 50, r: 20, t: 10, b: 40 },
            font: {
              family: 'JetBrains Mono, monospace',
              color: '#9ca3af',
              size: 10,
            },
            legend: {
              bgcolor: 'rgba(17,24,39,0.8)',
              bordercolor: '#374151',
              borderwidth: 1,
              font: { size: 9, color: '#9ca3af' },
              x: 1,
              xanchor: 'right',
              y: 1,
            },
            xaxis: {
              title: {
                text: 'PC1 (divergence axis)',
                font: { size: 9, color: '#6b7280' },
              },
              gridcolor: '#1f2937',
              zerolinecolor: '#374151',
              tickfont: { size: 8, color: '#6b7280' },
              linecolor: '#374151',
            },
            yaxis: {
              title: {
                text: 'PC2 (superposition axis)',
                font: { size: 9, color: '#6b7280' },
              },
              gridcolor: '#1f2937',
              zerolinecolor: '#374151',
              tickfont: { size: 8, color: '#6b7280' },
              linecolor: '#374151',
            },
            hoverlabel: {
              bgcolor: '#111827',
              bordercolor: '#374151',
              font: { family: 'JetBrains Mono', size: 10, color: '#e5e7eb' },
            },
          } as Partial<Plotly.Layout>
        }
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </div>
  )
}
