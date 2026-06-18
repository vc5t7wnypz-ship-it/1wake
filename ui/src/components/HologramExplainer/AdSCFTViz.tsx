import React, { useRef, useEffect, useCallback } from 'react'
import { ParamSlider } from './ParamSlider'
import { InsightBox } from './InsightBox'

interface Params {
  blackHoleMass: number    // mass → radius, entropy, Hawking temp
  bulkDimension: number    // N dimension (N→∞ is strong coupling limit)
  cutoffRadius: number     // UV cutoff: how close to boundary
  coupling: number         // g²N: 't Hooft coupling
}

interface AdSCFTVizProps {
  params: Params
  onParamChange: (key: keyof Params, val: number) => void
}

// Poincaré disk geodesic: geodesic between two boundary points
function geodesicPoints(theta1: number, theta2: number, r: number, nPts = 80): [number, number][] {
  // In the Poincaré disk, geodesics between boundary points are circular arcs
  const x1 = r * Math.cos(theta1)
  const y1 = r * Math.sin(theta1)
  const x2 = r * Math.cos(theta2)
  const y2 = r * Math.sin(theta2)

  // Midpoint of the chord
  const mx = (x1 + x2) / 2
  const my = (y1 + y2) / 2
  // The geodesic arc center is on the perpendicular bisector of the chord
  // at distance sqrt(1 - mx²-my²) from midpoint (for boundary r=1)
  const perpDist = 1 - (mx ** 2 + my ** 2)
  if (perpDist < 0.001) {
    // Nearly diametrically opposite: straight line
    return Array.from({ length: nPts }, (_, i) => {
      const s = i / (nPts - 1)
      return [x1 + s * (x2 - x1), y1 + s * (y2 - y1)] as [number, number]
    })
  }

  // Arc center
  const norm = Math.sqrt(mx ** 2 + my ** 2)
  if (norm < 0.001) {
    return [[x1, y1], [x2, y2]]
  }
  const scale = (1 + mx ** 2 + my ** 2) / (2 * norm)
  const arcCx = (mx / norm) * scale
  const arcCy = (my / norm) * scale
  const arcR = Math.sqrt((arcCx - x1) ** 2 + (arcCy - y1) ** 2)

  const a1 = Math.atan2(y1 - arcCy, x1 - arcCx)
  const a2 = Math.atan2(y2 - arcCy, x2 - arcCx)

  let da = a2 - a1
  if (da > Math.PI) da -= 2 * Math.PI
  if (da < -Math.PI) da += 2 * Math.PI

  return Array.from({ length: nPts }, (_, i) => {
    const s = i / (nPts - 1)
    const angle = a1 + s * da
    return [arcCx + arcR * Math.cos(angle), arcCy + arcR * Math.sin(angle)] as [number, number]
  })
}

export const AdSCFTViz: React.FC<AdSCFTVizProps> = ({ params, onParamChange }) => {
  const bulkCanvasRef = useRef<HTMLCanvasElement>(null)
  const boundaryCanvasRef = useRef<HTMLCanvasElement>(null)
  const tRef = useRef(0)
  const rafRef = useRef<number>(0)
  const paramsRef = useRef(params)
  paramsRef.current = params

  const draw = useCallback(() => {
    const bc = bulkCanvasRef.current
    const bdc = boundaryCanvasRef.current
    if (!bc || !bdc) return

    const W = bc.width
    const H = bc.height
    const { blackHoleMass, bulkDimension, cutoffRadius, coupling } = paramsRef.current
    const t = tRef.current

    const diskR = Math.min(W, H) * 0.42
    const cx = W / 2
    const cy = H / 2

    // ─── Bulk: Poincaré disk (AdS₃ slice) ─────────────────────────────────
    {
      const ctx = bc.getContext('2d')!
      ctx.fillStyle = '#020617'
      ctx.fillRect(0, 0, W, H)

      // Disk background gradient (AdS curvature visualization)
      const diskGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, diskR)
      diskGrad.addColorStop(0, 'rgba(30,27,75,0.9)')
      diskGrad.addColorStop(0.5, 'rgba(15,23,42,0.9)')
      diskGrad.addColorStop(1, 'rgba(6,8,24,0.95)')
      ctx.beginPath()
      ctx.arc(cx, cy, diskR, 0, Math.PI * 2)
      ctx.fillStyle = diskGrad
      ctx.fill()

      // Curvature grid lines (hyperbolic)
      const nGridR = 8
      for (let i = 1; i < nGridR; i++) {
        const pr = (i / nGridR)  // Poincaré radius 0..1
        const pixR = (pr * diskR * (1 - cutoffRadius * 0.1))
        ctx.beginPath()
        ctx.arc(cx, cy, pixR, 0, Math.PI * 2)
        ctx.strokeStyle = `rgba(30,58,138,${0.3 - i * 0.02})`
        ctx.lineWidth = 0.8
        ctx.stroke()
      }

      // Geodesics (RT surfaces)
      const nGeodesics = 5
      for (let i = 0; i < nGeodesics; i++) {
        const th1 = (i / nGeodesics) * Math.PI * 2
        const th2 = th1 + Math.PI * 0.7
        const pts = geodesicPoints(th1, th2, 0.99)

        ctx.beginPath()
        pts.forEach(([px, py], j) => {
          const sx = cx + px * diskR
          const sy = cy + py * diskR
          if (j === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        const alpha = 0.15 + 0.05 * Math.sin(t * 0.02 + i)
        ctx.strokeStyle = `rgba(99,102,241,${alpha})`
        ctx.lineWidth = 0.8
        ctx.stroke()
      }

      // Black hole in the center
      const bhR_raw = blackHoleMass * diskR * 0.25
      const bhR = Math.max(5, Math.min(diskR * 0.65, bhR_raw))

      // Ergosphere
      const ergoR = bhR * 1.35
      const ergoGrad = ctx.createRadialGradient(cx, cy, bhR, cx, cy, ergoR)
      ergoGrad.addColorStop(0, 'rgba(239,68,68,0.4)')
      ergoGrad.addColorStop(1, 'rgba(239,68,68,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, ergoR, 0, Math.PI * 2)
      ctx.fillStyle = ergoGrad
      ctx.fill()

      // Event horizon
      ctx.beginPath()
      ctx.arc(cx, cy, bhR, 0, Math.PI * 2)
      const bhGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, bhR)
      bhGrad.addColorStop(0, 'rgba(0,0,0,1)')
      bhGrad.addColorStop(0.8, 'rgba(10,5,30,1)')
      bhGrad.addColorStop(1, 'rgba(239,68,68,0.8)')
      ctx.fillStyle = bhGrad
      ctx.fill()
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 1.5
      ctx.stroke()

      // Hawking radiation particles (animated)
      const T_H = (1 / (8 * Math.PI * blackHoleMass + 0.01)) * coupling * 0.5
      const nParticles = Math.round(T_H * 20)
      for (let p = 0; p < nParticles; p++) {
        const angle = (p / nParticles) * Math.PI * 2 + t * 0.03
        const radius = bhR + (bhR * 0.5) * ((t * 0.5 + p * 17) % (diskR - bhR)) / (diskR - bhR)
        const px = cx + radius * Math.cos(angle)
        const py = cy + radius * Math.sin(angle)
        const fade = 1 - (radius - bhR) / (diskR - bhR)
        ctx.beginPath()
        ctx.arc(px, py, 1.5, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(251,191,36,${fade * 0.8})`
        ctx.fill()
      }

      // Entanglement wedge (RT minimal surface)
      const rtAngle1 = Math.PI * 0.1
      const rtAngle2 = Math.PI * 0.9
      const rtPts = geodesicPoints(rtAngle1, rtAngle2, 0.99)
      ctx.beginPath()
      rtPts.forEach(([px, py], j) => {
        const sx = cx + px * diskR
        const sy = cy + py * diskR
        if (j === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.strokeStyle = 'rgba(34,197,94,0.7)'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 3])
      ctx.stroke()
      ctx.setLineDash([])

      // RT label
      const rtMid = rtPts[Math.floor(rtPts.length / 2)]
      ctx.font = '10px monospace'
      ctx.fillStyle = '#22c55e'
      ctx.fillText('RT surface', cx + rtMid[0] * diskR + 6, cy + rtMid[1] * diskR)

      // Boundary circle
      ctx.beginPath()
      ctx.arc(cx, cy, diskR, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(56,189,248,0.6)'
      ctx.lineWidth = 2
      ctx.stroke()

      // Boundary label
      ctx.font = '11px monospace'
      ctx.fillStyle = 'rgba(56,189,248,0.8)'
      ctx.fillText('CFT boundary (d-1)', cx - 70, cy - diskR - 6)

      // Entropy formula
      const S_BH = (4 * Math.PI * blackHoleMass ** 2 * bulkDimension).toFixed(1)
      ctx.fillStyle = 'rgba(148,163,184,0.8)'
      ctx.font = '11px monospace'
      ctx.fillText(`S = A/4Gₙ ≈ ${S_BH}`, cx - 40, cy + diskR + 16)

      // Axis label
      ctx.fillStyle = '#475569'
      ctx.font = '10px monospace'
      ctx.fillText('AdS bulk (d+1 dim)', 6, H - 8)
    }

    // ─── Boundary: CFT degrees of freedom ─────────────────────────────────
    {
      const ctx = bdc.getContext('2d')!
      const BW = bdc.width, BH = bdc.height
      ctx.fillStyle = '#030712'
      ctx.fillRect(0, 0, BW, BH)

      // Draw 1D boundary as a circle
      const bcx = BW / 2, bcy = BH / 2
      const bR = Math.min(BW, BH) * 0.38

      // CFT operators as colored dots on the boundary
      const N = Math.round(bulkDimension)
      const nOps = N * 3

      ctx.beginPath()
      ctx.arc(bcx, bcy, bR, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(56,189,248,0.4)'
      ctx.lineWidth = 1.5
      ctx.stroke()

      // Entanglement entropy arcs (bipartite: region A and B)
      const aStart = Math.PI * 0.1
      const aEnd = Math.PI * 0.9

      // Region A (boundary interval)
      ctx.beginPath()
      ctx.arc(bcx, bcy, bR + 3, aStart, aEnd)
      ctx.strokeStyle = 'rgba(34,197,94,0.8)'
      ctx.lineWidth = 4
      ctx.stroke()

      ctx.font = '10px monospace'
      ctx.fillStyle = '#22c55e'
      const midA = (aStart + aEnd) / 2
      ctx.fillText('Region A', bcx + (bR + 14) * Math.cos(midA) - 25, bcy + (bR + 14) * Math.sin(midA))

      // Region B
      ctx.beginPath()
      ctx.arc(bcx, bcy, bR + 3, aEnd, aStart + 2 * Math.PI)
      ctx.strokeStyle = 'rgba(239,68,68,0.5)'
      ctx.lineWidth = 4
      ctx.stroke()

      // Operator dots
      for (let i = 0; i < nOps; i++) {
        const angle = (i / nOps) * Math.PI * 2 + t * 0.005
        const px = bcx + bR * Math.cos(angle)
        const py = bcy + bR * Math.sin(angle)

        // Color by entanglement: green=high, red=low
        const entangle = 0.5 + 0.4 * Math.sin(angle * 3 + t * 0.02)
        const r = Math.round(entangle * 200)
        const g = Math.round((1 - entangle) * 150 + 50)
        const b = 200

        ctx.beginPath()
        ctx.arc(px, py, 2.5, 0, Math.PI * 2)
        ctx.fillStyle = `rgb(${r},${g},${b})`
        ctx.fill()
      }

      // Entanglement web inside boundary
      const webPts = Array.from({ length: 12 }, (_, i) => {
        const angle = (i / 12) * Math.PI * 2
        return [bcx + bR * Math.cos(angle), bcy + bR * Math.sin(angle)] as [number, number]
      })

      for (let i = 0; i < webPts.length; i += 2) {
        const [x1, y1] = webPts[i]
        const [x2, y2] = webPts[(i + 5) % webPts.length]
        const alpha = 0.1 + 0.05 * Math.sin(t * 0.02 + i)
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        // Curved line through center
        ctx.quadraticCurveTo(bcx + (Math.random() - 0.5) * 20, bcy + (Math.random() - 0.5) * 20, x2, y2)
        ctx.strokeStyle = `rgba(99,102,241,${alpha})`
        ctx.lineWidth = 0.8
        ctx.stroke()
      }

      // 't Hooft coupling and central charge labels
      const c_charge = (coupling * N ** 2 / (12 * Math.PI)).toFixed(1)
      ctx.font = '11px monospace'
      ctx.fillStyle = 'rgba(148,163,184,0.8)'
      ctx.fillText(`λ = g²N = ${coupling.toFixed(1)}`, 8, 16)
      ctx.fillText(`c = ${c_charge}`, 8, 30)
      ctx.fillText(`N = ${N}`, 8, 44)

      ctx.fillStyle = '#475569'
      ctx.font = '10px monospace'
      ctx.fillText('CFT (boundary, d dim)', 6, BH - 8)

      // S_EE formula
      const S_EE = (coupling * N ** 2 / (12 * Math.PI) * Math.log(cutoffRadius * 100 + 2)).toFixed(2)
      ctx.fillStyle = 'rgba(34,197,94,0.8)'
      ctx.font = '10px monospace'
      ctx.fillText(`S_EE(A) = ${S_EE}`, BW - 100, BH - 8)
    }

    tRef.current += 1
    rafRef.current = requestAnimationFrame(draw)
  }, [])

  useEffect(() => {
    const canvases = [bulkCanvasRef.current, boundaryCanvasRef.current]
    const ros: ResizeObserver[] = []
    canvases.forEach((c) => {
      if (!c) return
      const ro = new ResizeObserver(() => {
        const r = c.getBoundingClientRect()
        c.width = r.width || 340
        c.height = r.height || 300
      })
      ro.observe(c)
      const r = c.getBoundingClientRect()
      c.width = r.width || 340
      c.height = r.height || 300
      ros.push(ro)
    })
    rafRef.current = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(rafRef.current)
      ros.forEach((r) => r.disconnect())
    }
  }, [draw])

  const bhArea = (4 * Math.PI * params.blackHoleMass ** 2).toFixed(2)
  const T_H = (1 / (8 * Math.PI * (params.blackHoleMass + 0.01))).toFixed(4)
  const S_BH = (Math.PI * params.blackHoleMass ** 2 * params.bulkDimension).toFixed(1)

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-mono text-gray-500 text-center">AdS bulk (d+1 dimensions)</span>
          <div className="rounded-xl overflow-hidden border border-indigo-900/50" style={{ height: 300 }}>
            <canvas ref={bulkCanvasRef} className="w-full h-full block" />
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-xs font-mono text-gray-500 text-center">CFT boundary (d dimensions)</span>
          <div className="rounded-xl overflow-hidden border border-cyan-900/50" style={{ height: 300 }}>
            <canvas ref={boundaryCanvasRef} className="w-full h-full block" />
          </div>
        </div>
      </div>

      {/* Live physics readouts */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Black hole entropy', value: `S = ${S_BH}`, formula: 'S = A/4Gₙ', color: 'text-red-400', border: 'border-red-900/40', bg: 'bg-red-950/20' },
          { label: 'Hawking temperature', value: `T = ${T_H}`, formula: 'T = 1/8πM', color: 'text-amber-400', border: 'border-amber-900/40', bg: 'bg-amber-950/20' },
          { label: 'Horizon area', value: `A = ${bhArea}`, formula: '= 4πM²', color: 'text-violet-400', border: 'border-violet-900/40', bg: 'bg-violet-950/20' },
        ].map(({ label, value, formula, color, border, bg }) => (
          <div key={label} className={`rounded-lg border ${border} ${bg} p-3 text-center`}>
            <div className={`font-mono text-lg font-bold ${color}`}>{value}</div>
            <div className="font-mono text-xs text-gray-500 mt-1">{formula}</div>
            <div className="text-xs text-gray-400 mt-1">{label}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        <ParamSlider label="Black hole mass M" value={params.blackHoleMass} min={0.1} max={2} step={0.05}
          onChange={(v) => onParamChange('blackHoleMass', v)} color="rose" />
        <ParamSlider label="N (gauge group rank)" value={params.bulkDimension} min={2} max={20} step={1}
          onChange={(v) => onParamChange('bulkDimension', v)} color="violet" />
        <ParamSlider label="UV cutoff ε" value={params.cutoffRadius} min={0.01} max={0.5} step={0.01}
          onChange={(v) => onParamChange('cutoffRadius', v)} color="sky" />
        <ParamSlider label="'t Hooft coupling λ=g²N" value={params.coupling} min={0.1} max={10} step={0.1}
          onChange={(v) => onParamChange('coupling', v)} color="emerald" />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <InsightBox icon="📐" title="Ryu-Takayanagi formula" variant="emerald">
          <p>The entanglement entropy S(A) of a boundary region A equals the area of the minimal geodesic surface (green curve) in the bulk that is homologous to A, divided by 4Gₙ. This is <strong className="text-emerald-300">exactly</strong> the Bekenstein-Hawking formula for black holes, applied to any region.</p>
        </InsightBox>
        <InsightBox icon="🌌" title="AdS/CFT duality" variant="violet">
          <p>A <strong className="text-violet-300">(d+1)-dimensional</strong> gravity theory in Anti-de Sitter space is exactly equivalent to a <strong className="text-violet-300">d-dimensional</strong> conformal field theory on its boundary — with no gravity. The universe may be a hologram encoded on a lower-dimensional surface.</p>
        </InsightBox>
      </div>
    </div>
  )
}
