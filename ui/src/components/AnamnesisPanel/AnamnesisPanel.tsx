import React, { useState } from 'react'
import { usePassageStore } from '../../store/passageStore'
import type { AnamnesisResult } from '../../store/passageStore'

const BAR_COLORS = [
  '#3b82f6', '#8b5cf6', '#f59e0b', '#22c55e', '#ef4444',
  '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#a78bfa',
]

const LensBar: React.FC<{ name: string; score: number; color: string }> = ({
  name,
  score,
  color,
}) => (
  <div className="flex items-center gap-2 text-xs font-mono">
    <div className="w-28 text-right text-gray-400 truncate">{name}</div>
    <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{
          width: `${Math.min(100, Math.round(score * 100))}%`,
          background: color,
          opacity: 0.85,
        }}
      />
    </div>
    <div className="w-10 text-gray-500 text-right">{score.toFixed(3)}</div>
  </div>
)

interface AnamnesisContentProps {
  anamnesis: AnamnesisResult
}

const AnamnesisContent: React.FC<AnamnesisContentProps> = ({ anamnesis }) => {
  return (
    <div className="flex flex-col gap-5 px-5 py-4">
      {/* Section 1: Implicit Query State */}
      {anamnesis.retrievedLenses.length > 0 && (
        <section>
          <h4 className="text-xs font-mono font-semibold text-gray-400 uppercase tracking-widest mb-3">
            Implicit Query State
          </h4>
          <div className="flex flex-col gap-1.5">
            {anamnesis.retrievedLenses.map(([name, score], i) => (
              <LensBar
                key={name}
                name={name}
                score={score}
                color={BAR_COLORS[i % BAR_COLORS.length]}
              />
            ))}
          </div>
        </section>
      )}

      {/* Section 2: Activated Graph Nodes */}
      {anamnesis.retrievedNodes.length > 0 && (
        <section>
          <h4 className="text-xs font-mono font-semibold text-gray-400 uppercase tracking-widest mb-3">
            Activated Graph Nodes
          </h4>
          <div className="flex flex-wrap gap-2">
            {anamnesis.retrievedNodes.slice(0, 20).map((node, i) => (
              <span
                key={i}
                className="chip"
                title={node.meaning ?? node.surface}
              >
                <span
                  className="mr-1.5 inline-block w-1.5 h-1.5 rounded-full"
                  style={{
                    background:
                      node.type === 'MythFigure'
                        ? '#a855f7'
                        : node.type === 'KabbalahNode'
                          ? '#eab308'
                          : node.type === 'SemanticField'
                            ? '#f59e0b'
                            : node.type === 'Root'
                              ? '#22c55e'
                              : '#3b82f6',
                  }}
                />
                {node.surface}
                <span className="ml-1.5 text-gray-600">
                  {node.score.toFixed(2)}
                </span>
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Section 3: Novel Connections */}
      {anamnesis.novelConnections.length > 0 && (
        <section>
          <h4 className="text-xs font-mono font-semibold text-gray-400 uppercase tracking-widest mb-3">
            Novel Connections
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-gray-600 border-b border-gray-800">
                  <th className="text-left py-1.5 pr-4 font-normal">from</th>
                  <th className="text-left py-1.5 pr-4 font-normal">to</th>
                  <th className="text-left py-1.5 pr-4 font-normal">path</th>
                  <th className="text-right py-1.5 font-normal">novelty</th>
                </tr>
              </thead>
              <tbody>
                {anamnesis.novelConnections.map((conn, i) => (
                  <tr
                    key={i}
                    className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
                  >
                    <td className="py-1.5 pr-4 text-sky-400">{conn.from}</td>
                    <td className="py-1.5 pr-4 text-purple-400">{conn.to}</td>
                    <td className="py-1.5 pr-4 text-gray-500">
                      {Array.isArray(conn.path) ? conn.path.join(' → ') : conn.path}
                    </td>
                    <td className="py-1.5 text-right">
                      <span
                        className={`
                          px-1.5 py-0.5 rounded text-xs
                          ${conn.noveltyScore > 0.7 ? 'bg-yellow-900/40 text-yellow-300' : 'bg-gray-800 text-gray-400'}
                        `}
                      >
                        {conn.noveltyScore.toFixed(3)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Section 4: Narrative */}
      {anamnesis.narrative && (
        <section>
          <h4 className="text-xs font-mono font-semibold text-gray-400 uppercase tracking-widest mb-3">
            Narrative
          </h4>
          <p className="text-sm text-gray-300 italic leading-relaxed border-l-2 border-sky-800 pl-3">
            {anamnesis.narrative}
          </p>
        </section>
      )}
    </div>
  )
}

export const AnamnesisPanel: React.FC = () => {
  const { result } = usePassageStore()
  const [open, setOpen] = useState(false)

  const anamnesis = result?.anamnesis

  return (
    <div
      className={`
        border-t border-gray-800 bg-gray-900/80
        transition-all duration-300 ease-in-out
      `}
    >
      {/* Header / toggle */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-3 px-5 py-2.5
                   hover:bg-gray-800/40 transition-colors duration-150"
      >
        <span
          className={`
            text-gray-500 text-xs transition-transform duration-200
            ${open ? 'rotate-90' : 'rotate-0'}
          `}
        >
          ▶
        </span>
        <span className="font-mono text-xs font-semibold uppercase tracking-widest text-gray-400">
          ANAMNESIS
        </span>
        <span className="text-gray-600 font-mono text-xs">
          — What does this passage want to be read by?
        </span>
        {anamnesis && (
          <span className="ml-auto chip text-yellow-400 border-yellow-700/50">
            {anamnesis.retrievedLenses.length} lenses · {anamnesis.retrievedNodes.length} nodes
          </span>
        )}
      </button>

      {/* Collapsible body */}
      <div
        className={`
          overflow-hidden transition-all duration-300 ease-in-out
          ${open ? 'max-h-[600px] opacity-100' : 'max-h-0 opacity-0'}
        `}
      >
        {anamnesis ? (
          <AnamnesisContent anamnesis={anamnesis} />
        ) : (
          <div className="px-5 py-4 text-xs font-mono text-gray-600">
            {result
              ? 'No anamnesis data returned by this analysis.'
              : 'Submit a passage to see anamnesis results.'}
          </div>
        )}
      </div>
    </div>
  )
}
