import React, { useRef, useEffect, useCallback } from 'react'
import { ParamSlider } from './ParamSlider'
import { InsightBox } from './InsightBox'

interface Params {
  focalLength: number   // focal length in pixels
  wavelength: number    // λ in arbitrary units
  numZones: number      // how many Fresnel zones to show
  offset3D: number      // virtual depth offset (simulates 3D parallax)
}

interface ZonePlateVizProps {
  params: Params
  onParamChange: (key: keyof Params, val: number) => void
}

export const ZonePlateViz: React.FC<ZonePlateVizProps> = ({ params, onParamChange }) => {
  const plateCanvasRef = useRef<HTMLCanvasElement>(null)
  const focusCanvasRef = useRef<HTMLCanvasElement>(null)
  const tRef = useRef(0)
  const rafRef = useRef<number>(0)
  const paramsRef = useRef(params)
  paramsRef.current = params

  const draw = useCallback(() => {
    const pc = plateCanvasRef.current
    const fc = focusCanvasRef.current
    if (!pc || !fc) return

    const W = pc.width
    const H = pc.height
    const { focalLength, wavelength, numZones, offset3D } = paramsRef.current
    const t = tRef.current
    const lambda = wavelength
    const f = focalLength

    // ─── Zone plate pattern ────────────────────────────────────────────────
    {
      const ctx = pc.getContext('2d')!
      const cx = W / 2, cy = H / 2
      const img = ctx.createImageData(W, H)

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const r2 = (x - cx) ** 2 + (y - cy) ** 2
          // Gabor zone plate: t(r) = 0.5 + 0.5*cos(π*r²/(λ*f))
          const phase = (Math.PI * r2) / (lambda * f * 3)
          const zone = Math.cos(phase)

          // Binary zone plate (alternating black/white rings)
          const bin = zone > 0 ? 1 : 0

          // Number of zones check
          const zoneNum = Math.floor(r2 / (lambda * f * 3 / Math.PI))
          const inRange = zoneNum < numZones * 6

          const b = inRange ? bin : 0

          const idx = (y * W + x) * 4
          img.data[idx]     = Math.round(b * 220)
          img.data[idx + 1] = Math.round(b * 220)
          img.data[idx + 2] = Math.round(b * 220)
          img.data[idx + 3] = 255
        }
      }
      ctx.putImageData(img, 0, 0)

      // Overlay animated phase ring
      const animR = Math.sqrt(lambda * f * 3 * t * 0.003) % (W / 2)
      ctx.beginPath()
      ctx.arc(cx, cy, animR, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(56,189,248,0.5)'
      ctx.lineWidth = 1.5
      ctx.stroke()

      ctx.font = '11px monospace'
      ctx.fillStyle = 'rgba(148,163,184,0.8)'
      ctx.fillText('Gabor Zone Plate', 8, H - 8)

      // Focal spot
      ctx.beginPath()
      ctx.arc(cx, cy, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#38bdf8'
      ctx.fill()
    }

    // ─── Depth/focus visualization ─────────────────────────────────────────
    {
      const ctx = fc.getContext('2d')!
      ctx.fillStyle = '#030712'
      ctx.fillRect(0, 0, W, H)

      const plateX = W * 0.3

      // Incident wave (from left)
      for (let wavX = 0; wavX < plateX; wavX += 20) {
        const alpha = 0.1
        ctx.strokeStyle = `rgba(147,197,253,${alpha})`
        ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(wavX, 0); ctx.lineTo(wavX, H); ctx.stroke()
      }

      // Converging beams
      const nRays = 16
      const f_scaled = focalLength * 0.8
      const focalPt = plateX + f_scaled
      const off3D = offset3D * 60

      for (let i = 0; i < nRays; i++) {
        const sy = (i / (nRays - 1)) * H
        // Three focal points: near, at-focus, far (3D depth cues)
        for (const [fPtX, fPtY, alpha, color] of [
          [focalPt, H / 2 + off3D, 0.35, '56,189,248'],
          [focalPt * 0.6, H / 2, 0.12, '148,163,184'],
          [focalPt * 1.6, H / 2 - off3D * 0.5, 0.12, '148,163,184'],
        ] as [number, number, number, string][]) {
          const phase = Math.sqrt((fPtX - plateX) ** 2 + (sy - fPtY) ** 2) * 0.1
          const a = alpha * Math.max(0, Math.cos(phase - t * 0.06))
          if (a < 0.01) continue

          const grad = ctx.createLinearGradient(plateX, sy, fPtX, fPtY)
          grad.addColorStop(0, `rgba(${color},${a})`)
          grad.addColorStop(1, `rgba(${color},0)`)
          ctx.strokeStyle = grad
          ctx.lineWidth = 0.8
          ctx.beginPath()
          ctx.moveTo(plateX, sy)
          ctx.lineTo(fPtX, fPtY)
          ctx.stroke()
        }
      }

      // Draw plate
      ctx.fillStyle = '#1e293b'
      ctx.fillRect(plateX - 4, 0, 8, H)
      ctx.strokeStyle = '#475569'
      ctx.lineWidth = 1
      ctx.strokeRect(plateX - 4, 0, 8, H)

      // Focal spot glow
      const glowX = focalPt
      const glowY = H / 2 + off3D
      const glow = ctx.createRadialGradient(glowX, glowY, 0, glowX, glowY, 25 + 8 * Math.sin(t * 0.05))
      glow.addColorStop(0, 'rgba(56,189,248,0.7)')
      glow.addColorStop(0.3, 'rgba(56,189,248,0.3)')
      glow.addColorStop(1, 'rgba(56,189,248,0)')
      ctx.fillStyle = glow
      ctx.beginPath()
      ctx.arc(glowX, glowY, 30, 0, Math.PI * 2)
      ctx.fill()

      // Labels
      ctx.font = '10px monospace'
      ctx.fillStyle = '#64748b'
      ctx.fillText('zone plate', plateX - 34, H - 8)
      ctx.fillStyle = '#38bdf8'
      ctx.fillText('focus', glowX + 6, glowY - 8)
      ctx.fillStyle = '#64748b'
      ctx.fillText(`f = ${focalLength.toFixed(0)}px`, glowX + 6, glowY + 18)

      // Depth arrow
      if (Math.abs(off3D) > 5) {
        ctx.fillStyle = '#fbbf24'
        ctx.font = '10px monospace'
        ctx.fillText('depth offset', glowX + 6, glowY + 32)
        ctx.strokeStyle = '#fbbf24'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(glowX, H / 2)
        ctx.lineTo(glowX, glowY)
        ctx.stroke()
      }
    }

    tRef.current += 1
    rafRef.current = requestAnimationFrame(draw)
  }, [])

  useEffect(() => {
    const canvases = [plateCanvasRef.current, focusCanvasRef.current]
    const ros: ResizeObserver[] = []
    canvases.forEach((c) => {
      if (!c) return
      const ro = new ResizeObserver(() => {
        const r = c.getBoundingClientRect()
        c.width = r.width || 300
        c.height = r.height || 260
      })
      ro.observe(c)
      const r = c.getBoundingClientRect()
      c.width = r.width || 300
      c.height = r.height || 260
      ros.push(ro)
    })
    rafRef.current = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(rafRef.current)
      ros.forEach((r) => r.disconnect())
    }
  }, [draw])

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-mono text-gray-500 text-center">Zone plate pattern</span>
          <div className="rounded-xl overflow-hidden border border-gray-800 bg-black" style={{ height: 260 }}>
            <canvas ref={plateCanvasRef} className="w-full h-full block" />
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-xs font-mono text-gray-500 text-center">Diffraction & focus</span>
          <div className="rounded-xl overflow-hidden border border-gray-800" style={{ height: 260 }}>
            <canvas ref={focusCanvasRef} className="w-full h-full block" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        <ParamSlider label="Focal length" value={params.focalLength} min={30} max={180} step={5}
          unit="px" onChange={(v) => onParamChange('focalLength', v)} color="sky" />
        <ParamSlider label="Wavelength" value={params.wavelength} min={5} max={40} step={1}
          onChange={(v) => onParamChange('wavelength', v)} color="violet" />
        <ParamSlider label="Number of zones" value={params.numZones} min={1} max={20} step={1}
          onChange={(v) => onParamChange('numZones', v)} color="emerald" />
        <ParamSlider label="3D depth offset" value={params.offset3D} min={-1} max={1} step={0.05}
          onChange={(v) => onParamChange('offset3D', v)} color="amber" />
      </div>

      <InsightBox icon="⬭" title="Gabor's zone plate = lens + hologram" variant="emerald">
        <p>Dennis Gabor (Nobel 1971) realized that a simple pattern of concentric rings — called a <strong className="text-emerald-300">zone plate</strong> — can focus light like a lens through pure diffraction. The rings record the interference between a plane wave and a spherical wave from a point source.</p>
        <p className="mt-2">Unlike a lens, a zone plate can reconstruct multiple focal depths simultaneously. The <strong className="text-emerald-300">3D depth offset</strong> slider shows how shifting the recorded phase moves the focal point in 3D space — the hologram encodes genuine depth information, not just a 2D image.</p>
      </InsightBox>
    </div>
  )
}
