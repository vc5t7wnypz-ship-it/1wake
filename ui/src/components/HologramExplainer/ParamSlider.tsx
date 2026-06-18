import React from 'react'

interface ParamSliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  unit?: string
  onChange: (v: number) => void
  color?: string
}

export const ParamSlider: React.FC<ParamSliderProps> = ({
  label, value, min, max, step = 0.01, unit = '', onChange, color = 'sky'
}) => {
  const pct = ((value - min) / (max - min)) * 100

  const trackColor: Record<string, string> = {
    sky: 'from-sky-600 to-cyan-400',
    violet: 'from-violet-600 to-fuchsia-400',
    emerald: 'from-emerald-600 to-teal-400',
    amber: 'from-amber-600 to-yellow-400',
    rose: 'from-rose-600 to-pink-400',
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-baseline">
        <span className="text-xs font-mono text-gray-400">{label}</span>
        <span className={`text-xs font-mono font-bold text-${color}-400`}>
          {Number.isInteger(step) || step >= 1 ? Math.round(value) : value.toFixed(2)}
          {unit && <span className="text-gray-500 ml-0.5">{unit}</span>}
        </span>
      </div>
      <div className="relative h-6 flex items-center">
        <div className="absolute inset-x-0 h-1.5 rounded-full bg-gray-800 overflow-hidden">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${trackColor[color] ?? trackColor.sky} transition-none`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div
          className={`absolute w-3.5 h-3.5 rounded-full bg-${color}-400 border-2 border-gray-950
                      shadow-[0_0_8px_rgba(56,189,248,0.6)] pointer-events-none transition-none`}
          style={{ left: `calc(${pct}% - 7px)` }}
        />
      </div>
    </div>
  )
}
