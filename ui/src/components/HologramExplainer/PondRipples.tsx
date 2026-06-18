import React, { useRef, useEffect, useCallback } from 'react'
import { ParamSlider } from './ParamSlider'
import { InsightBox } from './InsightBox'

interface Params {
  wavelength: number   // pixels between wavefronts
  speed: number        // omega = 2π/period
  separation: number   // distance between two sources (fraction of canvas width)
  amplitude: number
}

interface PondRipplesProps {
  params: Params
  onParamChange: (key: keyof Params, val: number) => void
}

export const PondRipples: React.FC<PondRipplesProps> = ({ params, onParamChange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tRef = useRef(0)
  const rafRef = useRef<number>(0)
  const paramsRef = useRef(params)
  paramsRef.current = params

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    const { wavelength, speed, separation, amplitude } = paramsRef.current
    const t = tRef.current

    const k = (2 * Math.PI) / wavelength
    const omega = speed * 0.05

    const cx = W / 2
    const cy = H / 2
    const sep = separation * W * 0.35

    const s1x = cx - sep / 2
    const s1y = cy
    const s2x = cx + sep / 2
    const s2y = cy

    const imgData = ctx.createImageData(W, H)
    const data = imgData.data

    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const r1 = Math.sqrt((px - s1x) ** 2 + (py - s1y) ** 2)
        const r2 = Math.sqrt((px - s2x) ** 2 + (py - s2y) ** 2)

        const decay1 = amplitude / Math.max(1, Math.sqrt(r1) * 0.3)
        const decay2 = amplitude / Math.max(1, Math.sqrt(r2) * 0.3)

        const w1 = decay1 * Math.sin(k * r1 - omega * t)
        const w2 = decay2 * Math.sin(k * r2 - omega * t)
        const sum = (w1 + w2) * 0.5

        // Map to color: positive → cyan/blue, negative → dark navy
        const norm = Math.max(-1, Math.min(1, sum))
        const bright = (norm + 1) / 2  // 0..1

        const idx = (py * W + px) * 4
        if (bright > 0.5) {
          // constructive: bright cyan
          const b = (bright - 0.5) * 2
          data[idx]     = Math.round(b * 30)
          data[idx + 1] = Math.round(b * 180)
          data[idx + 2] = Math.round(b * 220)
          data[idx + 3] = 255
        } else {
          // destructive: dark blue
          const b = bright * 2
          data[idx]     = Math.round(b * 5)
          data[idx + 1] = Math.round(b * 15)
          data[idx + 2] = Math.round(b * 60)
          data[idx + 3] = 255
        }
      }
    }

    ctx.putImageData(imgData, 0, 0)

    // Draw source dots
    for (const [sx, sy] of [[s1x, s1y], [s2x, s2y]] as [number, number][]) {
      ctx.beginPath()
      ctx.arc(sx, sy, 5, 0, Math.PI * 2)
      ctx.fillStyle = '#f0f9ff'
      ctx.fill()
      ctx.beginPath()
      ctx.arc(sx, sy, 3, 0, Math.PI * 2)
      ctx.fillStyle = '#38bdf8'
      ctx.fill()
    }

    // Label sources
    ctx.font = '11px monospace'
    ctx.fillStyle = 'rgba(148,163,184,0.9)'
    ctx.fillText('Stone 1', s1x - 22, s1y + 20)
    ctx.fillText('Stone 2', s2x - 22, s2y + 20)

    tRef.current += 1
    rafRef.current = requestAnimationFrame(draw)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width
      canvas.height = rect.height
    })
    ro.observe(canvas)
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width || 500
    canvas.height = rect.height || 300
    rafRef.current = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(rafRef.current)
      ro.disconnect()
    }
  }, [draw])

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-xl overflow-hidden border border-gray-800" style={{ height: 280 }}>
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        <ParamSlider label="Wavelength" value={params.wavelength} min={15} max={80} step={1}
          unit="px" onChange={(v) => onParamChange('wavelength', v)} color="sky" />
        <ParamSlider label="Wave speed" value={params.speed} min={0.2} max={3} step={0.1}
          onChange={(v) => onParamChange('speed', v)} color="cyan" />
        <ParamSlider label="Stone separation" value={params.separation} min={0.05} max={0.9} step={0.01}
          onChange={(v) => onParamChange('separation', v)} color="violet" />
        <ParamSlider label="Amplitude" value={params.amplitude} min={0.5} max={3} step={0.1}
          onChange={(v) => onParamChange('amplitude', v)} color="emerald" />
      </div>

      <InsightBox icon="🌊" title="What you're seeing" variant="cyan">
        <p>Each stone creates circular ripples. Where two crests meet → <strong className="text-cyan-300">bright constructive interference</strong>. Where a crest meets a trough → <strong className="text-blue-400">dark destructive interference</strong>.</p>
        <p className="mt-2">The pattern of bright and dark lines <em>permanently records</em> information about where both stones were — even after the stones are gone. That pattern <em>is</em> a hologram.</p>
      </InsightBox>
    </div>
  )
}
