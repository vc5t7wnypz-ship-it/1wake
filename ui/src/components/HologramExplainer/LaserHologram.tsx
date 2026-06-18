import React, { useRef, useEffect, useCallback } from 'react'
import { ParamSlider } from './ParamSlider'
import { InsightBox } from './InsightBox'

interface Params {
  wavelength: number   // nm display, maps to k
  objectAngle: number  // angle of object beam in degrees
  objectDist: number   // distance of point object from film
  coherence: number    // coherence multiplier 0..1
}

interface LaserHologramProps {
  params: Params
  onParamChange: (key: keyof Params, val: number) => void
}

export const LaserHologram: React.FC<LaserHologramProps> = ({ params, onParamChange }) => {
  const recordCanvasRef = useRef<HTMLCanvasElement>(null)
  const recoCanvasRef = useRef<HTMLCanvasElement>(null)
  const tRef = useRef(0)
  const rafRef = useRef<number>(0)
  const paramsRef = useRef(params)
  paramsRef.current = params

  const draw = useCallback(() => {
    const rc = recordCanvasRef.current
    const rv = recoCanvasRef.current
    if (!rc || !rv) return

    const W = rc.width
    const H = rc.height
    const { wavelength, objectAngle, objectDist, coherence } = paramsRef.current
    const t = tRef.current

    // k proportional to (visible range 400-700nm)
    const k = (2 * Math.PI) / (wavelength * 0.08)  // scale to pixels

    // ─── Left canvas: Recording ───────────────────────────────────────────
    {
      const ctx = rc.getContext('2d')!
      ctx.fillStyle = '#030712'
      ctx.fillRect(0, 0, W, H)

      // Film strip on right side (x = W*0.7)
      const filmX = W * 0.68

      // Reference beam: plane wave traveling in +x direction
      // Object beam: spherical wave from point (objX, objY)
      const objAngleRad = (objectAngle * Math.PI) / 180
      const objX = filmX - objectDist * Math.cos(objAngleRad)
      const objY = H / 2 - objectDist * Math.sin(objAngleRad)

      // Draw reference beam (parallel lines moving right)
      const refPeriod = 18
      for (let x = 0; x < filmX; x += refPeriod) {
        const phase = k * x - t * 0.08
        const alpha = 0.12 + 0.08 * Math.sin(phase)
        ctx.strokeStyle = `rgba(147,197,253,${alpha})`
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, H)
        ctx.stroke()
      }

      // Draw object (small sphere)
      ctx.beginPath()
      ctx.arc(objX, objY, 8, 0, Math.PI * 2)
      const gObj = ctx.createRadialGradient(objX, objY, 0, objX, objY, 8)
      gObj.addColorStop(0, '#f0f9ff')
      gObj.addColorStop(1, '#7dd3fc')
      ctx.fillStyle = gObj
      ctx.fill()
      ctx.font = '10px monospace'
      ctx.fillStyle = '#94a3b8'
      ctx.fillText('object', objX + 12, objY + 4)

      // Draw object beam (circular waves from object)
      for (let r = 5; r < 300; r += 20) {
        const phase = k * r - t * 0.08
        const alpha = coherence * 0.15 * Math.max(0, Math.cos(phase))
        if (alpha < 0.01) continue
        ctx.beginPath()
        ctx.arc(objX, objY, r, 0, Math.PI * 2)
        ctx.strokeStyle = `rgba(251,191,36,${alpha})`
        ctx.lineWidth = 1
        ctx.stroke()
      }

      // Draw beam splitter
      const bsX = W * 0.25
      ctx.strokeStyle = 'rgba(148,163,184,0.5)'
      ctx.lineWidth = 2
      ctx.setLineDash([4, 3])
      ctx.beginPath()
      ctx.moveTo(bsX - 15, H / 2 - 20)
      ctx.lineTo(bsX + 15, H / 2 + 20)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.font = '10px monospace'
      ctx.fillStyle = '#64748b'
      ctx.fillText('beam splitter', bsX - 40, H / 2 - 30)

      // Laser source on left
      const laserGrad = ctx.createLinearGradient(10, H / 2 - 6, 10, H / 2 + 6)
      laserGrad.addColorStop(0, '#dc2626')
      laserGrad.addColorStop(1, '#f87171')
      ctx.fillStyle = laserGrad
      ctx.fillRect(8, H / 2 - 6, 20, 12)
      ctx.font = '10px monospace'
      ctx.fillStyle = '#94a3b8'
      ctx.fillText(`laser ${wavelength}nm`, 8, H / 2 - 12)

      // Film strip
      ctx.fillStyle = 'rgba(15,23,42,0.9)'
      ctx.fillRect(filmX - 4, 0, 12, H)
      ctx.strokeStyle = '#475569'
      ctx.lineWidth = 1
      ctx.strokeRect(filmX - 4, 0, 12, H)

      // Draw interference pattern ON the film strip
      const filmImgData = rc.getContext('2d')!.createImageData(8, H)
      for (let fy = 0; fy < H; fy++) {
        const filmY = fy
        // Reference beam phase at film
        const phaseRef = k * filmX

        // Object beam phase at film
        const rObj = Math.sqrt((filmX - objX) ** 2 + (filmY - objY) ** 2)
        const phaseObj = k * rObj

        // Interference intensity (time-averaged)
        const phaseDiff = phaseRef - phaseObj
        const coherFactor = coherence
        const I = 0.5 + 0.5 * coherFactor * Math.cos(phaseDiff)

        const brightness = Math.round(I * 200)
        const idx = fy * 4 * 8
        for (let fx = 0; fx < 8; fx++) {
          filmImgData.data[idx + fx * 4]     = Math.round(brightness * 0.15)
          filmImgData.data[idx + fx * 4 + 1] = Math.round(brightness * 0.70)
          filmImgData.data[idx + fx * 4 + 2] = Math.round(brightness * 0.90)
          filmImgData.data[idx + fx * 4 + 3] = 255
        }
      }
      rc.getContext('2d')!.putImageData(filmImgData, filmX - 4, 0)

      ctx.font = '10px monospace'
      ctx.fillStyle = '#64748b'
      ctx.fillText('holographic film', filmX - 40, H - 8)

      // Arrow labels
      ctx.fillStyle = '#7dd3fc'
      ctx.font = '10px monospace'
      ctx.fillText('ref. beam →', bsX + 20, H / 2 - 6)
    }

    // ─── Right canvas: Reconstruction ─────────────────────────────────────
    {
      const ctx = rv.getContext('2d')!
      const RW = rv.width
      const RH = rv.height

      ctx.fillStyle = '#030712'
      ctx.fillRect(0, 0, RW, RH)

      const { objectAngle, objectDist, coherence } = paramsRef.current
      const filmX = RW * 0.25

      // Draw reference beam illuminating the film
      for (let x = 0; x < filmX + 10; x += 18) {
        const alpha = 0.1 + 0.06 * Math.sin(k * x - t * 0.08)
        ctx.strokeStyle = `rgba(147,197,253,${alpha})`
        ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, RH); ctx.stroke()
      }

      // Reconstructed virtual object
      const objAngleRad = (objectAngle * Math.PI) / 180
      const virObjX = filmX + objectDist * Math.cos(objAngleRad) * 0.8
      const virObjY = RH / 2 - objectDist * Math.sin(objAngleRad) * 0.8

      // Diffracted beams converging to virtual image
      const nBeams = 12
      for (let i = 0; i < nBeams; i++) {
        const fy = (i / (nBeams - 1)) * RH
        const rToObj = Math.sqrt((virObjX - filmX) ** 2 + (fy - virObjY) ** 2)
        const phDiff = k * rToObj

        const alpha = coherence * 0.25 * Math.max(0, Math.cos(phDiff - t * 0.05))
        if (alpha < 0.01) continue

        const grad = ctx.createLinearGradient(filmX, fy, virObjX, virObjY)
        grad.addColorStop(0, `rgba(251,191,36,${alpha})`)
        grad.addColorStop(1, `rgba(251,191,36,0)`)
        ctx.strokeStyle = grad
        ctx.lineWidth = 1.2
        ctx.beginPath()
        ctx.moveTo(filmX, fy)
        ctx.lineTo(virObjX, virObjY)
        ctx.stroke()
      }

      // Virtual image glow
      const glowR = 20 + 5 * Math.sin(t * 0.04)
      const gObj = ctx.createRadialGradient(virObjX, virObjY, 0, virObjX, virObjY, glowR)
      gObj.addColorStop(0, `rgba(251,191,36,${0.3 * coherence})`)
      gObj.addColorStop(0.4, `rgba(251,191,36,${0.15 * coherence})`)
      gObj.addColorStop(1, 'rgba(251,191,36,0)')
      ctx.fillStyle = gObj
      ctx.beginPath()
      ctx.arc(virObjX, virObjY, glowR, 0, Math.PI * 2)
      ctx.fill()

      // Object dot
      ctx.beginPath()
      ctx.arc(virObjX, virObjY, 6, 0, Math.PI * 2)
      ctx.fillStyle = '#fde68a'
      ctx.fill()
      ctx.font = '10px monospace'
      ctx.fillStyle = '#fde68a'
      ctx.fillText('virtual image', virObjX + 10, virObjY - 8)

      // Film strip
      ctx.fillStyle = '#1e293b'
      ctx.fillRect(filmX - 4, 0, 10, RH)
      ctx.strokeStyle = '#475569'
      ctx.lineWidth = 1
      ctx.strokeRect(filmX - 4, 0, 10, RH)
      ctx.font = '10px monospace'
      ctx.fillStyle = '#64748b'
      ctx.fillText('developed film', filmX - 42, RH - 8)

      // Eyes
      const eyeX = RW - 40
      const eyeY = RH / 2
      ctx.font = '20px serif'
      ctx.fillText('👁', eyeX, eyeY + 7)
      ctx.font = '9px monospace'
      ctx.fillStyle = '#94a3b8'
      ctx.fillText('observer', eyeX - 10, eyeY + 24)
    }

    tRef.current += 1
    rafRef.current = requestAnimationFrame(draw)
  }, [])

  useEffect(() => {
    const canvases = [recordCanvasRef.current, recoCanvasRef.current]
    const ros: ResizeObserver[] = []
    canvases.forEach((c) => {
      if (!c) return
      const ro = new ResizeObserver(() => {
        const r = c.getBoundingClientRect()
        c.width = r.width || 400
        c.height = r.height || 220
      })
      ro.observe(c)
      const r = c.getBoundingClientRect()
      c.width = r.width || 400
      c.height = r.height || 220
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
          <span className="text-xs font-mono text-gray-500 px-1">Recording</span>
          <div className="rounded-xl overflow-hidden border border-gray-800" style={{ height: 220 }}>
            <canvas ref={recordCanvasRef} className="w-full h-full block" />
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-xs font-mono text-gray-500 px-1">Reconstruction</span>
          <div className="rounded-xl overflow-hidden border border-gray-800" style={{ height: 220 }}>
            <canvas ref={recoCanvasRef} className="w-full h-full block" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        <ParamSlider label="Laser wavelength" value={params.wavelength} min={400} max={700} step={5}
          unit="nm" onChange={(v) => onParamChange('wavelength', v)} color="violet" />
        <ParamSlider label="Object angle" value={params.objectAngle} min={-60} max={60} step={1}
          unit="°" onChange={(v) => onParamChange('objectAngle', v)} color="amber" />
        <ParamSlider label="Object distance" value={params.objectDist} min={40} max={200} step={5}
          unit="px" onChange={(v) => onParamChange('objectDist', v)} color="sky" />
        <ParamSlider label="Coherence" value={params.coherence} min={0.05} max={1} step={0.05}
          onChange={(v) => onParamChange('coherence', v)} color="emerald" />
      </div>

      <InsightBox icon="🔬" title="Why lasers?" variant="violet">
        <p>Ordinary light is <em>incoherent</em> — photons arrive randomly and can't maintain a stable phase relationship. A laser emits light where all photons share the same phase, wavelength, and direction, making the interference fringes on the film sharp and stable.</p>
        <p className="mt-2">Lower the <strong className="text-violet-300">coherence</strong> slider to see the hologram blur and disappear — this is why holograms don't work under room light.</p>
      </InsightBox>
    </div>
  )
}
