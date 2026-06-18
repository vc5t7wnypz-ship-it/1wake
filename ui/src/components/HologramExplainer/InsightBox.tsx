import React from 'react'

interface InsightBoxProps {
  icon?: string
  title: string
  children: React.ReactNode
  variant?: 'cyan' | 'violet' | 'emerald' | 'amber'
}

const variantStyles = {
  cyan:    { border: 'border-cyan-800/60',   bg: 'bg-cyan-950/40',   title: 'text-cyan-300',   icon: 'text-cyan-400' },
  violet:  { border: 'border-violet-800/60', bg: 'bg-violet-950/40', title: 'text-violet-300', icon: 'text-violet-400' },
  emerald: { border: 'border-emerald-800/60',bg: 'bg-emerald-950/40',title: 'text-emerald-300',icon: 'text-emerald-400' },
  amber:   { border: 'border-amber-800/60',  bg: 'bg-amber-950/40',  title: 'text-amber-300',  icon: 'text-amber-400' },
}

export const InsightBox: React.FC<InsightBoxProps> = ({
  icon = '◆', title, children, variant = 'cyan'
}) => {
  const s = variantStyles[variant]
  return (
    <div className={`rounded-lg border ${s.border} ${s.bg} p-4 text-sm`}>
      <div className={`flex items-center gap-2 font-mono font-bold mb-2 ${s.title}`}>
        <span className={s.icon}>{icon}</span>
        {title}
      </div>
      <div className="text-gray-300 leading-relaxed">{children}</div>
    </div>
  )
}
