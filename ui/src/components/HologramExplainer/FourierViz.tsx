import React, { useRef, useEffect, useCallback } from 'react'
import { ParamSlider } from './ParamSlider'
import { InsightBox } from './InsightBox'

interface Params {
  numFreqs: number      // number of frequency components shown
  maskRadius: number    // how much of Fourier space is visible (0=none,1=all)
  phaseShift: number    // global phase shift in degrees
  noiseLevel: number    // noise added to the pattern
}

interface FourierVizProps {
  params: Params
  onParamChange: (key: keyof Params, val: number) => void
}

// Compute a simple 2D DFT magnitude for display (down-sampled)
function buildFourierPattern(W: number, H: number, numFreqs: number, noise: number): Float32Array {
  const data = new Float32Array(W * H)
  const cx = W / 2
  const cy = H / 2

  // Sum several sinusoids
  const freqs = [
    { kx: 0.08, ky: 0.05, amp: 1.0 },
    { kx: 0.03, ky: 0.12, amp: 0.7 },
    { kx: 0.15, ky: 0.03, amp: 0.5 },
    { kx: 0.06, ky: 0.18, amp: 0.4 },
    { kx: 0.20, ky: 0.10, amp: 0.3 },
    { kx: 0.02, ky: 0.25, amp: 0.25 },
    { kx: 0.30, ky: 0.08, amp: 0.2 },
    { kx: 0.12, ky: 0.22, amp: 0.18 },
  ].slice(0, numFreqs)

  let max = 0
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let v = 0
      for (const { kx, ky, amp } of freqs) {
        v += amp * Math.cos(2 * Math.PI * (kx * (x - cx) + ky * (y - cy)))
      }
      // Add noise
      v += noise * (Math.random() * 2 - 1) * 0.5
      data[y * W + x] = v
      if (Math.abs(v) > max) max = Math.abs(v)
    }
  }
  // Normalize
  for (let i = 0; i < data.length; i++) data[i] = data[i] / max
  return data
}

function buildFreqDomain(W: number, H: number, numFreqs: number, maskR: number, phase: number): Float32Array {
  const data = new Float32Array(W * H)
  const cx = W / 2
  const cy = H / 2
  const phRad = (phase * Math.PI) / 180

  const freqs = [
    { kx: 0.08, ky: 0.05, amp: 1.0 },
    { kx: 0.03, ky: 0.12, amp: 0.7 },
    { kx: 0.15, ky: 0.03, amp: 0.5 },
    { kx: 0.06, ky: 0.18, amp: 0.4 },
    { kx: 0.20, ky: 0.10, amp: 0.3 },
    { kx: 0.02, ky: 0.25, amp: 0.25 },
    { kx: 0.30, ky: 0.08, amp: 0.2 },
    { kx: 0.12, ky: 0.22, amp: 0.18 },
  ].slice(0, numFreqs)

  const maxK = Math.sqrt(0.35 ** 2 + 0.35 ** 2)

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const fx = (x - cx) / W
      const fy = (y - cy) / H
      const r = Math.sqrt(fx ** 2 + fy ** 2)

      // Is this frequency inside the mask?
      const masked = r > maskR * maxK

      let v = 0
      for (const { kx, ky, amp } of freqs) {
        // Dot each freq component as a bright spot + its conjugate
        const sigma = 0.005
        const d1 = Math.sqrt((fx - kx) ** 2 + (fy - ky) ** 2)
        const d2 = Math.sqrt((fx + kx) ** 2 + (fy + ky) ** 2)
        v += amp * (Math.exp(-(d1 * d1) / sigma) + Math.exp(-(d2 * d2) / sigma))
          * Math.cos(phRad) * (masked ? 0 : 1)
      }
      data[y * W + x] = v
    }
  }
  return data
}

export const FourierViz: React.FC<FourierVizProps> = ({ params, onParamChange }) => {
  const spatialRef = useRef<HTMLCanvasElement>(null)
  const freqRef = useRef<HTMLCanvasElement>(null)
  const recoRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const tRef = useRef(0)
  const paramsRef = useRef(params)
  paramsRef.current = params

  const draw = useCallback(() => {
    const sc = spatialRef.current
    const fc = freqRef.current
    const rc = recoRef.current
    if (!sc || !fc || !rc) return

    const W = sc.width
    const H = sc.height
    const { numFreqs, maskRadius, phaseShift, noiseLevel } = paramsRef.current
    const t = tRef.current

    // ─── Spatial domain: original pattern ──────────────────────────────────
    {
      const ctx = sc.getContext('2d')!
      const pattern = buildFourierPattern(W, H, numFreqs, noiseLevel)
      const img = ctx.createImageData(W, H)
      for (let i = 0; i < W * H; i++) {
        const v = pattern[i]
        const b = (v + 1) / 2
        img.data[i * 4]     = Math.round(b * 30)
        img.data[i * 4 + 1] = Math.round(b * 160)
        img.data[i * 4 + 2] = Math.round(b * 200)
        img.data[i * 4 + 3] = 255
      }
      ctx.putImageData(img, 0, 0)
      ctx.font = '11px monospace'
      ctx.fillStyle = 'rgba(148,163,184,0.8)'
      ctx.fillText('Object (spatial domain)', 8, H - 8)
    }

    // ─── Frequency domain: Fourier transform ───────────────────────────────
    {
      const ctx = fc.getContext('2d')!
      ctx.fillStyle = '#030712'
      ctx.fillRect(0, 0, W, H)

      const freq = buildFreqDomain(W, H, numFreqs, maskRadius, phaseShift)
      const img = ctx.createImageData(W, H)
      for (let i = 0; i < W * H; i++) {
        const v = Math.min(1, freq[i] * 0.3)
        img.data[i * 4]     = Math.round(v * 220)
        img.data[i * 4 + 1] = Math.round(v * 200)
        img.data[i * 4 + 2] = Math.round(v * 60)
        img.data[i * 4 + 3] = 255
      }
      ctx.putImageData(img, 0, 0)

      // Draw mask circle
      const cx = W / 2, cy = H / 2
      const maxK = Math.sqrt(0.35 ** 2 + 0.35 ** 2)
      const maskPx = maskRadius * maxK * W

      ctx.beginPath()
      ctx.arc(cx, cy, maskPx, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(239,68,68,0.6)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([5, 3])
      ctx.stroke()
      ctx.setLineDash([])

      // Animate phase: rotating phase arrow at center
      const arrowAngle = (phaseShift * Math.PI) / 180 + t * 0.02
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.lineTo(cx + 12 * Math.cos(arrowAngle), cy + 12 * Math.sin(arrowAngle))
      ctx.strokeStyle = 'rgba(252,211,77,0.9)'
      ctx.lineWidth = 2
      ctx.stroke()

      ctx.font = '11px monospace'
      ctx.fillStyle = 'rgba(148,163,184,0.8)'
      ctx.fillText('Fourier space', 8, H - 8)
    }

    // ─── Reconstruction: inverse FT with mask applied ──────────────────────
    {
      const ctx = rc.getContext('2d')!
      const img = ctx.createImageData(W, H)
      // maskRadius controls how many freqs survive reconstruction
      const nVisible = Math.round(numFreqs * maskRadius)
      const rPattern = buildFourierPattern(W, H, Math.max(1, nVisible), 0)
      for (let i = 0; i < W * H; i++) {
        const v = rPattern[i]
        const b = (v + 1) / 2
        // Fade based on info preserved
        const fidelity = maskRadius
        const gray = b * (1 - fidelity)
        img.data[i * 4]     = Math.round(gray * 80 + b * fidelity * 30)
        img.data[i * 4 + 1] = Math.round(gray * 80 + b * fidelity * 160)
        img.data[i * 4 + 2] = Math.round(gray * 80 + b * fidelity * 200)
        img.data[i * 4 + 3] = 255
      }
      ctx.putImageData(img, 0, 0)
      ctx.font = '11px monospace'
      ctx.fillStyle = 'rgba(148,163,184,0.8)'
      ctx.fillText(`Reconstructed (${Math.round(maskRadius * 100)}% freqs)`, 8, H - 8)
    }

    tRef.current += 1
    rafRef.current = requestAnimationFrame(draw)
  }, [])

  useEffect(() => {
    const canvases = [spatialRef.current, freqRef.current, recoRef.current]
    const ros: ResizeObserver[] = []
    canvases.forEach((c) => {
      if (!c) return
      const ro = new ResizeObserver(() => {
        const r = c.getBoundingClientRect()
        c.width = r.width || 180
        c.height = r.height || 180
      })
      ro.observe(c)
      const r = c.getBoundingClientRect()
      c.width = r.width || 180
      c.height = r.height || 180
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
      <div className="grid grid-cols-3 gap-3">
        {[
          { ref: spatialRef, label: 'Spatial' },
          { ref: freqRef, label: 'Fourier Space' },
          { ref: recoRef, label: 'Reconstruction' },
        ].map(({ ref, label }) => (
          <div key={label} className="flex flex-col gap-1">
            <span className="text-xs font-mono text-gray-500 text-center">{label}</span>
            <div className="rounded-xl overflow-hidden border border-gray-800 aspect-square">
              <canvas ref={ref} className="w-full h-full block" />
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        <ParamSlider label="# frequency components" value={params.numFreqs} min={1} max={8} step={1}
          onChange={(v) => onParamChange('numFreqs', v)} color="amber" />
        <ParamSlider label="Fourier mask radius" value={params.maskRadius} min={0.05} max={1} step={0.01}
          onChange={(v) => onParamChange('maskRadius', v)} color="rose" />
        <ParamSlider label="Phase shift" value={params.phaseShift} min={0} max={360} step={5}
          unit="°" onChange={(v) => onParamChange('phaseShift', v)} color="violet" />
        <ParamSlider label="Noise" value={params.noiseLevel} min={0} max={2} step={0.05}
          onChange={(v) => onParamChange('noiseLevel', v)} color="sky" />
      </div>

      <InsightBox icon="∿" title="Non-local information storage" variant="amber">
        <p>Every point in Fourier space encodes information from the <em>entire</em> spatial pattern. Shrink the <strong className="text-amber-300">mask radius</strong> — you lose high-frequency detail first (like blur), but structure survives because the low-frequency content is still present.</p>
        <p className="mt-2">A holographic film works the same way: damage any part of the film and you lose some resolution globally, but never lose a region of the image entirely. This is fundamentally different from a photograph.</p>
      </InsightBox>
    </div>
  )
}
