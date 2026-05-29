import React from 'react'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const sizeMap: Record<string, string> = {
  sm: 'w-4 h-4 border-2',
  md: 'w-6 h-6 border-2',
  lg: 'w-10 h-10 border-4',
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  className = '',
}) => {
  return (
    <span
      className={`
        inline-block rounded-full
        border-gray-700 border-t-sky-400
        animate-spin
        ${sizeMap[size] ?? sizeMap.md}
        ${className}
      `}
      role="status"
      aria-label="Loading"
    />
  )
}
