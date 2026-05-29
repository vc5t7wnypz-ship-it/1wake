import React, { useState, useEffect } from 'react'
import { usePassageStore } from '../../store/passageStore'
import { LoadingSpinner } from '../common/LoadingSpinner'

export const PassageInput: React.FC = () => {
  const {
    isLoading,
    availableLenses,
    selectedLenses,
    setSelectedLenses,
    analyzePassage,
    loadAvailableLenses,
  } = usePassageStore()

  const [passage, setPassage] = useState(
    "riverrun, past Eve and Adam's, from swerve of shore to bend of bay",
  )
  const [page, setPage] = useState(3)
  const [line, setLine] = useState(1)

  useEffect(() => {
    loadAvailableLenses()
  }, [loadAvailableLenses])

  const toggleLens = (lens: string) => {
    if (selectedLenses.includes(lens)) {
      if (selectedLenses.length > 1) {
        setSelectedLenses(selectedLenses.filter((l) => l !== lens))
      }
    } else {
      setSelectedLenses([...selectedLenses, lens])
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!passage.trim() || isLoading) return
    analyzePassage(passage.trim(), page, line, selectedLenses)
  }

  const LENS_COLORS: Record<string, string> = {
    mythological: 'border-purple-500/60 text-purple-300',
    kabbalistic: 'border-yellow-500/60 text-yellow-300',
    psychoanalytic: 'border-blue-500/60 text-blue-300',
    historical: 'border-green-500/60 text-green-300',
    linguistic: 'border-orange-500/60 text-orange-300',
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3 w-full">
      {/* Passage textarea */}
      <div className="flex gap-3 items-start">
        <textarea
          value={passage}
          onChange={(e) => setPassage(e.target.value)}
          placeholder="riverrun, past Eve and Adam's..."
          rows={2}
          className="input-field flex-1 resize-none text-sm leading-relaxed"
          disabled={isLoading}
          aria-label="Passage text"
        />

        {/* Page / Line */}
        <div className="flex flex-col gap-1.5">
          <label className="flex items-center gap-2 text-xs font-mono text-gray-400">
            <span className="w-8">pg.</span>
            <input
              type="number"
              value={page}
              onChange={(e) => setPage(Number(e.target.value))}
              min={1}
              className="input-field w-16 py-1 text-center"
              disabled={isLoading}
              aria-label="Page number"
            />
          </label>
          <label className="flex items-center gap-2 text-xs font-mono text-gray-400">
            <span className="w-8">ln.</span>
            <input
              type="number"
              value={line}
              onChange={(e) => setLine(Number(e.target.value))}
              min={1}
              className="input-field w-16 py-1 text-center"
              disabled={isLoading}
              aria-label="Line number"
            />
          </label>
        </div>
      </div>

      {/* Lens checkboxes + submit */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs font-mono text-gray-500 uppercase tracking-widest">
          Lenses:
        </span>
        {availableLenses.map((lens) => {
          const active = selectedLenses.includes(lens)
          const colorCls =
            LENS_COLORS[lens] ?? 'border-gray-500/60 text-gray-300'
          return (
            <label
              key={lens}
              className={`
                flex items-center gap-1.5 cursor-pointer select-none
                px-2 py-0.5 rounded-full border text-xs font-mono
                transition-all duration-150
                ${active ? colorCls + ' bg-gray-800/60' : 'border-gray-700 text-gray-600 bg-transparent'}
              `}
            >
              <input
                type="checkbox"
                checked={active}
                onChange={() => toggleLens(lens)}
                disabled={isLoading}
                className="sr-only"
              />
              <span
                className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-current' : 'bg-gray-700'}`}
              />
              {lens}
            </label>
          )
        })}

        <div className="ml-auto">
          <button
            type="submit"
            disabled={isLoading || !passage.trim()}
            className="btn-primary"
          >
            {isLoading ? (
              <>
                <LoadingSpinner size="sm" />
                Analyzing…
              </>
            ) : (
              <>
                <span className="text-sky-300 text-base leading-none">◈</span>
                Analyze
              </>
            )}
          </button>
        </div>
      </div>
    </form>
  )
}
