import React, { useRef, useEffect, useCallback } from 'react'
import { ParamSlider } from './ParamSlider'
import { InsightBox } from './InsightBox'

interface Params {
  refractiveIndex: number   // n of glass (1.0=air, 1.5=glass, 1.8=diamond)
  curvature: number         // curvature of wine glass rim profile
  lightAngle: number        // incident beam angle in degrees
  wavelengthMix: number     // 0=monochromatic, 1=full spectrum (chromatic aberration)
}

interface CausticsVizProps {
  params: Params
  onParamChange: (key: keyof Params, val: number) => void
}

// Wine glass profile: returns (x, y) pairs for the glass cross-section
function glassProfile(H: number, curvature: number, cy: number): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = []
  const steps = 60

  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    const y = cy + H * 0.25 - t * H * 0.55
    // Wine glass profile: bowl widens from stem, curved sides
    const stemWidth = 4
    const rimWidth = H * 0.22
    const bowlCurve = Math.pow(t, curvature) * rimWidth + stemWidth
    pts.push({ x: bowlCurve, y })
  }
  return pts
}

// Snell's law: returns refracted angle (or NaN for total internal reflection)
function snell(theta_i: number, n1: number, n2: number): number {
  const sinT = (n1 / n2) * Math.sin(theta_i)
  if (Math.abs(sinT) > 1) return NaN  // TIR
  return Math.asin(sinT)
}

export const CausticsViz: React.FC<CausticsVizProps> = ({ params, onParamChange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tRef = useRef(0)
  const rafRef = useRef<number>(0)
  const paramsRef = useRef(params)
  paramsRef.current = params

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const W = canvas.width
    const H = canvas.height
    const { refractiveIndex, curvature, lightAngle, wavelengthMix } = paramsRef.current
    const t = tRef.current

    ctx.fillStyle = '#010b1a'
    ctx.fillRect(0, 0, W, H)

    const cx = W * 0.5
    const cy = H * 0.38
    const scaleH = H * 0.7

    const profile = glassProfile(scaleH, curvature, cy)
    const lightAngleRad = (lightAngle * Math.PI) / 180

    // ─── Collect caustic intensity on a floor ──────────────────────────────
    const floorY = cy + scaleH * 0.32
    const rimY = cy - scaleH * 0.30
    const floorIntensity = new Float32Array(W).fill(0)

    // Wavelength colors for dispersion
    const wavelengths = wavelengthMix > 0.1
      ? [
          { nm: 450, color: [100, 120, 255], n: refractiveIndex + 0.04 * wavelengthMix },
          { nm: 530, color: [80, 220, 80], n: refractiveIndex + 0.01 * wavelengthMix },
          { nm: 620, color: [255, 100, 60], n: refractiveIndex - 0.02 * wavelengthMix },
        ]
      : [{ nm: 550, color: [180, 220, 255], n: refractiveIndex }]

    // ─── Draw rays ─────────────────────────────────────────────────────────
    const nRays = 48
    const rayStartY = cy - scaleH * 0.55 - 40

    for (const { color, n } of wavelengths) {
      for (let ri = 0; ri < nRays; ri++) {
        const rayX0 = cx - W * 0.35 + (ri / (nRays - 1)) * W * 0.7

        // Find where this ray hits the glass surface
        // Approximate glass surface as smooth curve at x = ±profile[j].x
        let hitX = -1, hitY = -1, hitNormal = 0

        for (let j = 1; j < profile.length; j++) {
          const py0 = profile[j - 1].y
          const py1 = profile[j].y
          const px = profile[j].x + cx  // right side of glass

          if (rayX0 < cx && py0 >= rayStartY && py1 < rayStartY) continue

          // Ray travels at angle lightAngleRad from vertical
          // Ray: (rayX0 + s*sin(lightAngleRad), rayStartY + s*cos(lightAngleRad))
          // Find s where y = py1
          if (py1 >= rayStartY && py0 <= rayStartY) continue

          const s = (py1 - rayStartY) / Math.cos(lightAngleRad + 0.001)
          const rx = rayX0 + s * Math.sin(lightAngleRad)

          // Check if ray hits the right side of glass
          if (Math.abs(rx - px) < 6 && py1 > rimY && py1 < floorY) {
            hitX = rx
            hitY = py1

            // Normal direction: tangent of profile = (profile[j].x-profile[j-1].x, py1-py0)
            const tx = profile[j].x - profile[j - 1].x
            const ty = py1 - py0
            const tLen = Math.sqrt(tx ** 2 + ty ** 2)
            hitNormal = Math.atan2(-tx / tLen, ty / tLen)  // outward normal
            break
          }
        }

        if (hitX < 0) {
          // Ray doesn't hit glass: draw straight to floor
          const rayAlpha = 0.06
          ctx.beginPath()
          ctx.moveTo(rayX0, rayStartY)
          const fx = rayX0 + (floorY - rayStartY) * Math.tan(lightAngleRad)
          ctx.lineTo(fx, floorY)
          ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},${rayAlpha})`
          ctx.lineWidth = 0.7
          ctx.stroke()

          const fx_clamp = Math.max(0, Math.min(W - 1, Math.round(fx)))
          floorIntensity[fx_clamp] += 0.3
          continue
        }

        // Draw incoming ray
        ctx.beginPath()
        ctx.moveTo(rayX0, rayStartY)
        ctx.lineTo(hitX, hitY)
        ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},0.12)`
        ctx.lineWidth = 0.7
        ctx.stroke()

        // Refraction at entry surface
        const incidentAngle = Math.abs(lightAngleRad - hitNormal)
        const refractedAngle = snell(incidentAngle, 1.0, n)
        if (isNaN(refractedAngle)) continue

        // Refracted direction in glass
        const sign = hitX > cx ? -1 : 1
        const refractDir = sign * (lightAngleRad + (refractedAngle - incidentAngle) * 0.4)

        // Find exit point on bottom of glass (simplified: travel to just past center)
        const exitY = hitY + (floorY - hitY) * 0.85
        const exitX = hitX + (exitY - hitY) * Math.tan(refractDir)

        // Draw refracted ray inside glass (subtle)
        ctx.beginPath()
        ctx.moveTo(hitX, hitY)
        ctx.lineTo(exitX, exitY)
        ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},0.06)`
        ctx.lineWidth = 0.6
        ctx.stroke()

        // Exit refraction
        const exitAngle = snell(Math.abs(refractedAngle), n, 1.0)
        if (isNaN(exitAngle)) continue

        const exitDir = lightAngleRad + (exitAngle - incidentAngle) * sign * 0.6

        // Draw exit ray to floor
        const fx = exitX + (floorY - exitY) * Math.tan(exitDir)
        ctx.beginPath()
        ctx.moveTo(exitX, exitY)
        ctx.lineTo(fx, floorY)
        ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},0.22)`
        ctx.lineWidth = 0.8
        ctx.stroke()

        // Accumulate floor intensity
        const fx_clamp = Math.max(0, Math.min(W - 1, Math.round(fx)))
        for (let dx = -4; dx <= 4; dx++) {
          const ix = fx_clamp + dx
          if (ix >= 0 && ix < W) {
            floorIntensity[ix] += Math.exp(-(dx ** 2) / 6)
          }
        }
      }
    }

    // ─── Draw glass shape ──────────────────────────────────────────────────
    // Glass fill (semi-transparent)
    ctx.beginPath()
    ctx.moveTo(cx + profile[0].x, profile[0].y)
    for (const { x, y } of profile) ctx.lineTo(cx + x, y)
    for (let i = profile.length - 1; i >= 0; i--) ctx.lineTo(cx - profile[i].x, profile[i].y)
    ctx.closePath()
    ctx.fillStyle = 'rgba(147,210,255,0.06)'
    ctx.fill()
    ctx.strokeStyle = 'rgba(147,210,255,0.35)'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Glass stem
    ctx.beginPath()
    ctx.moveTo(cx - 4, cy + scaleH * 0.25)
    ctx.lineTo(cx - 4, cy + scaleH * 0.42)
    ctx.lineTo(cx + 4, cy + scaleH * 0.42)
    ctx.lineTo(cx + 4, cy + scaleH * 0.25)
    ctx.strokeStyle = 'rgba(147,210,255,0.3)'
    ctx.lineWidth = 1.2
    ctx.stroke()

    // Base
    ctx.beginPath()
    ctx.moveTo(cx - H * 0.1, cy + scaleH * 0.42)
    ctx.lineTo(cx + H * 0.1, cy + scaleH * 0.42)
    ctx.strokeStyle = 'rgba(147,210,255,0.3)'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // ─── Draw caustic on floor ─────────────────────────────────────────────
    const maxI = Math.max(...Array.from(floorIntensity))

    for (let x = 0; x < W; x++) {
      const norm = floorIntensity[x] / (maxI + 0.001)
      const h = norm * 30

      // Chromatic caustic color
      const r = wavelengthMix > 0.1 ? Math.round(norm * 255) : Math.round(norm * 230)
      const g = Math.round(norm * 220)
      const b = wavelengthMix > 0.1 ? Math.round(norm * 150) : Math.round(norm * 255)

      const grd = ctx.createLinearGradient(x, floorY, x, floorY + h)
      grd.addColorStop(0, `rgba(${r},${g},${b},${norm * 0.9})`)
      grd.addColorStop(1, `rgba(${r},${g},${b},0)`)
      ctx.fillStyle = grd
      ctx.fillRect(x, floorY, 1, h)
    }

    // Floor line
    ctx.beginPath()
    ctx.moveTo(0, floorY)
    ctx.lineTo(W, floorY)
    ctx.strokeStyle = 'rgba(51,65,85,0.8)'
    ctx.lineWidth = 1
    ctx.stroke()

    // ─── Animated light source ─────────────────────────────────────────────
    const srcY = rayStartY - 15
    ctx.beginPath()
    ctx.moveTo(0, srcY)
    ctx.lineTo(W, srcY)
    ctx.strokeStyle = `rgba(226,232,240,0.08)`
    ctx.lineWidth = 1
    ctx.stroke()

    // Incident light arrows
    for (let i = 0; i < 7; i++) {
      const ax = W * 0.1 + i * W * 0.12
      const ay = srcY + 8
      const ah = 18
      ctx.beginPath()
      ctx.moveTo(ax, ay)
      const ex = ax + ah * Math.sin(lightAngleRad)
      const ey = ay + ah * Math.cos(lightAngleRad)
      ctx.lineTo(ex, ey)
      const alpha = 0.4 + 0.1 * Math.sin(t * 0.05 + i)
      ctx.strokeStyle = `rgba(224,242,254,${alpha})`
      ctx.lineWidth = 1.2
      ctx.stroke()
      // Arrowhead
      ctx.beginPath()
      ctx.arc(ex, ey, 1.5, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(224,242,254,${alpha})`
      ctx.fill()
    }

    // Labels
    ctx.font = '11px monospace'
    ctx.fillStyle = 'rgba(148,163,184,0.8)'
    ctx.fillText(`n = ${refractiveIndex.toFixed(2)}`, 8, 16)
    ctx.fillText('caustic pattern', cx - 44, floorY + 48)
    ctx.fillText('incident light', 8, srcY - 4)

    // "Fold caustic" marker at bright spot
    let brightX = 0, brightVal = 0
    for (let x = 0; x < W; x++) {
      if (floorIntensity[x] > brightVal) { brightVal = floorIntensity[x]; brightX = x }
    }
    if (brightVal > 0.5) {
      ctx.beginPath()
      ctx.arc(brightX, floorY, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#fbbf24'
      ctx.fill()
      ctx.font = '10px monospace'
      ctx.fillStyle = '#fbbf24'
      ctx.fillText('cusp', brightX + 6, floorY - 4)
    }

    tRef.current += 1
    rafRef.current = requestAnimationFrame(draw)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(() => {
      const r = canvas.getBoundingClientRect()
      canvas.width = r.width || 600
      canvas.height = r.height || 400
    })
    ro.observe(canvas)
    const r = canvas.getBoundingClientRect()
    canvas.width = r.width || 600
    canvas.height = r.height || 400
    rafRef.current = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(rafRef.current)
      ro.disconnect()
    }
  }, [draw])

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-xl overflow-hidden border border-gray-800" style={{ height: 380 }}>
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        <ParamSlider label="Refractive index n" value={params.refractiveIndex} min={1.0} max={2.4} step={0.02}
          onChange={(v) => onParamChange('refractiveIndex', v)} color="sky" />
        <ParamSlider label="Glass curvature" value={params.curvature} min={0.5} max={3} step={0.05}
          onChange={(v) => onParamChange('curvature', v)} color="emerald" />
        <ParamSlider label="Incident light angle" value={params.lightAngle} min={-25} max={25} step={1}
          unit="°" onChange={(v) => onParamChange('lightAngle', v)} color="amber" />
        <ParamSlider label="Chromatic dispersion" value={params.wavelengthMix} min={0} max={1} step={0.05}
          onChange={(v) => onParamChange('wavelengthMix', v)} color="rose" />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <InsightBox icon="🍷" title="Caustics = focused interference" variant="cyan">
          <p>When light refracts through a curved glass surface (Snell's law: n₁ sin θ₁ = n₂ sin θ₂), rays converge onto <strong className="text-cyan-300">caustic surfaces</strong> — the bright wiggling patterns you see on a table under a wine glass.</p>
          <p className="mt-2">The bright <strong className="text-yellow-400">cusp point</strong> is a <em>catastrophe</em> in the mathematical sense: a fold singularity where infinite rays converge. The shape of the caustic <em>encodes the full 3D geometry</em> of the glass.</p>
        </InsightBox>
        <InsightBox icon="📡" title="Caustics → Holograms" variant="amber">
          <p>A caustic records geometry just like a hologram, but <em>lossy</em>. A true hologram records the full amplitude <em>and</em> phase of the wavefront, allowing perfect reconstruction. Increase the <strong className="text-amber-300">chromatic dispersion</strong> to see how different wavelengths separate — the holographic principle requires phase coherence to reconstruct faithfully.</p>
        </InsightBox>
      </div>
    </div>
  )
}
