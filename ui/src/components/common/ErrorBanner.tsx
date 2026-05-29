import React from 'react'
import { usePassageStore } from '../../store/passageStore'

interface ErrorBannerProps {
  message: string
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({ message }) => {
  const setError = usePassageStore((s) => s.setError)

  return (
    <div
      className="flex items-start gap-3 px-4 py-3 bg-red-950/60 border border-red-700/50
                 rounded-lg text-red-300 font-mono text-sm animate-fade-in"
      role="alert"
    >
      <span className="mt-0.5 text-red-400 flex-shrink-0 text-base leading-none">
        ⚠
      </span>
      <span className="flex-1 leading-relaxed">{message}</span>
      <button
        onClick={() => setError(null)}
        className="flex-shrink-0 text-red-500 hover:text-red-300 transition-colors
                   leading-none text-base font-bold ml-2"
        aria-label="Dismiss error"
      >
        ×
      </button>
    </div>
  )
}
