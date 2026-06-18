import React, { useState } from 'react'
import { PondRipples } from './PondRipples'
import { LaserHologram } from './LaserHologram'
import { CausticsViz } from './CausticsViz'
import { FourierViz } from './FourierViz'
import { ZonePlateViz } from './ZonePlateViz'
import { AdSCFTViz } from './AdSCFTViz'

// ─── Level definitions ────────────────────────────────────────────────────────

const LEVELS = [
  {
    id: 'eli5',
    label: 'ELI5',
    title: 'Two Stones in a Pond',
    sublabel: 'Explain Like I\'m 5',
    color: 'sky',
    gradient: 'from-sky-600 to-cyan-400',
    borderColor: 'border-sky-800/50',
    bgColor: 'bg-sky-950/20',
    tagBg: 'bg-sky-900/40',
    tagText: 'text-sky-300',
  },
  {
    id: 'highschool',
    label: 'High School',
    title: 'Laser Light & Wave Interference',
    sublabel: 'Recording & Reconstruction',
    color: 'violet',
    gradient: 'from-violet-600 to-fuchsia-400',
    borderColor: 'border-violet-800/50',
    bgColor: 'bg-violet-950/20',
    tagBg: 'bg-violet-900/40',
    tagText: 'text-violet-300',
  },
  {
    id: 'caustics',
    label: 'Intermediate',
    title: 'Caustics: Geometry in Light',
    sublabel: 'Wine glass · Snell\'s law · Catastrophes',
    color: 'amber',
    gradient: 'from-amber-600 to-yellow-400',
    borderColor: 'border-amber-800/50',
    bgColor: 'bg-amber-950/20',
    tagBg: 'bg-amber-900/40',
    tagText: 'text-amber-300',
  },
  {
    id: 'undergrad',
    label: 'Undergrad',
    title: 'Fourier Optics & Non-local Storage',
    sublabel: 'Every point encodes everything',
    color: 'emerald',
    gradient: 'from-emerald-600 to-teal-400',
    borderColor: 'border-emerald-800/50',
    bgColor: 'bg-emerald-950/20',
    tagBg: 'bg-emerald-900/40',
    tagText: 'text-emerald-300',
  },
  {
    id: 'grad',
    label: 'Graduate',
    title: 'Gabor Zone Plates & 3D Depth',
    sublabel: 'Fresnel zones · Coherence · Phase',
    color: 'rose',
    gradient: 'from-rose-600 to-pink-400',
    borderColor: 'border-rose-800/50',
    bgColor: 'bg-rose-950/20',
    tagBg: 'bg-rose-900/40',
    tagText: 'text-rose-300',
  },
  {
    id: 'phd',
    label: 'PhD',
    title: 'Holographic Principle & AdS/CFT',
    sublabel: 'Bekenstein-Hawking · Ryu-Takayanagi · ER=EPR',
    color: 'indigo',
    gradient: 'from-indigo-600 to-purple-500',
    borderColor: 'border-indigo-800/50',
    bgColor: 'bg-indigo-950/20',
    tagBg: 'bg-indigo-900/40',
    tagText: 'text-indigo-300',
  },
] as const

type LevelId = typeof LEVELS[number]['id']

// ─── Default parameter state for each level ───────────────────────────────────

const useAllParams = () => {
  const [eli5, setEli5] = useState({ wavelength: 35, speed: 1.2, separation: 0.45, amplitude: 1.5 })
  const [hs, setHs] = useState({ wavelength: 532, objectAngle: 20, objectDist: 100, coherence: 0.9 })
  const [caustics, setCaustics] = useState({ refractiveIndex: 1.5, curvature: 1.8, lightAngle: 0, wavelengthMix: 0 })
  const [ugrad, setUgrad] = useState({ numFreqs: 5, maskRadius: 0.9, phaseShift: 0, noiseLevel: 0 })
  const [grad, setGrad] = useState({ focalLength: 90, wavelength: 18, numZones: 12, offset3D: 0 })
  const [phd, setPhd] = useState({ blackHoleMass: 0.6, bulkDimension: 10, cutoffRadius: 0.05, coupling: 3.0 })

  const updater = <T extends Record<string, number>>(setter: React.Dispatch<React.SetStateAction<T>>) =>
    (key: keyof T, val: number) => setter((prev) => ({ ...prev, [key]: val }))

  return {
    eli5: { params: eli5, update: updater(setEli5) },
    highschool: { params: hs, update: updater(setHs) },
    caustics: { params: caustics, update: updater(setCaustics) },
    undergrad: { params: ugrad, update: updater(setUgrad) },
    grad: { params: grad, update: updater(setGrad) },
    phd: { params: phd, update: updater(setPhd) },
  }
}

// ─── Level content descriptions ───────────────────────────────────────────────

const LEVEL_PROSE: Record<LevelId, React.ReactNode> = {
  eli5: (
    <div className="space-y-3 text-sm text-gray-300 leading-relaxed">
      <p>
        Imagine throwing <strong className="text-sky-300">two stones</strong> into a perfectly calm pond at the same time.
        Each stone sends out circular ripples in all directions. When two ripple crests meet, they combine into
        a <strong className="text-cyan-300">big wave</strong> (constructive interference). When a crest meets a trough,
        they cancel out into <strong className="text-blue-400">stillness</strong> (destructive interference).
      </p>
      <p>
        The beautiful pattern of bright lines and dark lines that forms is called an <strong className="text-sky-300">interference pattern</strong>.
        Now here's the magical part: if you could instantly freeze the water surface and keep the frozen pattern forever,
        you'd have a <strong className="text-cyan-300">hologram</strong> — a physical record of <em>where both stones were</em>.
      </p>
      <p>
        Later, you could shine light through this frozen pattern and recreate the exact same ripple pattern — and it would
        look like the stones are still there in 3D, even though the water is gone.
      </p>
    </div>
  ),
  highschool: (
    <div className="space-y-3 text-sm text-gray-300 leading-relaxed">
      <p>
        Light is a wave — just like water ripples, but vibrating electromagnetic fields oscillating at ~500 trillion times per second.
        To record a hologram you need <strong className="text-violet-300">coherent light</strong> (a laser) where all photons share
        the same phase, wavelength, and direction.
      </p>
      <p>
        The laser beam is split into two paths: a <strong className="text-blue-300">reference beam</strong> (clean plane wave) and
        an <strong className="text-yellow-300">object beam</strong> that reflects off the 3D object. Both beams meet at a
        photographic plate and create a microscopic interference pattern — just like the pond, but at nanometer scale.
      </p>
      <p>
        To <strong className="text-violet-300">reconstruct</strong> the image, you simply shine the same reference beam through
        the developed film. The interference fringes bend the light back into the exact same wavefront that originally came off
        the object — and your eye sees a perfect 3D image floating in space.
      </p>
    </div>
  ),
  caustics: (
    <div className="space-y-3 text-sm text-gray-300 leading-relaxed">
      <p>
        Look at the shimmering light patterns on the bottom of a swimming pool, or on a table under a wine glass.
        These are <strong className="text-amber-300">caustics</strong> — light concentrated by curved surfaces through refraction.
        They're one of nature's most beautiful and mathematically rich phenomena.
      </p>
      <p>
        When light passes through a curved glass, <strong className="text-amber-300">Snell's law</strong> (n₁ sin θ₁ = n₂ sin θ₂)
        bends each ray by a different angle depending on where it hits. Rays from nearby parts of the surface converge to
        the same point: this convergence surface is the <strong className="text-yellow-300">caustic</strong>.
      </p>
      <p>
        The brightest points are <strong className="text-yellow-300">cusps</strong> — singularities from <em>catastrophe theory</em>
        (fold, cusp, swallowtail). Here, infinitely many rays pile up. The caustic pattern contains full information about
        the 3D geometry of the glass — a lossy proto-hologram encoded by geometry rather than phase.
      </p>
    </div>
  ),
  undergrad: (
    <div className="space-y-3 text-sm text-gray-300 leading-relaxed">
      <p>
        A photograph stores light intensity at each point — damage one area and that region's image is gone forever.
        A hologram is fundamentally different: it stores information in <strong className="text-emerald-300">Fourier space</strong>.
        Every point of the holographic film contains low-frequency contributions from the <em>entire</em> object.
      </p>
      <p>
        The Fourier transform decomposes any pattern into spatial frequencies. The hologram records the
        <strong className="text-emerald-300"> amplitude and phase</strong> of each frequency component across the film.
        Destroy part of the film and you lose some resolution, but every remaining piece still sees <em>all</em> of the
        object (at lower resolution) — non-local information storage.
      </p>
      <p>
        Use the <strong className="text-emerald-300">mask radius</strong> to see this in action: restricting to low frequencies
        gives a blurry but complete image. No part of the object disappears — the information is smeared everywhere.
        This is why broken holograms still work.
      </p>
    </div>
  ),
  grad: (
    <div className="space-y-3 text-sm text-gray-300 leading-relaxed">
      <p>
        Dennis Gabor (Nobel Prize 1971) developed holography while trying to improve electron microscopy.
        His key insight: any spherical wavefront from a point source interferes with a reference plane wave
        to create a <strong className="text-rose-300">zone plate</strong> — a bulls-eye pattern of alternating
        transparent and opaque rings.
      </p>
      <p>
        The zone plate is simultaneously a <strong className="text-rose-300">lens</strong> (it focuses light by diffraction)
        and a <strong className="text-pink-300">hologram</strong> (it records the phase information needed for 3D reconstruction).
        The radius of the n-th zone is rₙ = √(nλf), where f is focal length and λ is wavelength.
      </p>
      <p>
        The <strong className="text-rose-300">3D depth</strong> information is encoded in the chirp rate of the zone plate —
        how fast the rings get closer together toward the edge. Different depths produce different chirp rates,
        allowing a single flat film to store a complete 3D scene.
      </p>
    </div>
  ),
  phd: (
    <div className="space-y-3 text-sm text-gray-300 leading-relaxed">
      <p>
        In 1993 Gerard 't Hooft and Leonard Susskind proposed the <strong className="text-indigo-300">Holographic Principle</strong>:
        the maximum information content of any region of space is proportional to its <em>surface area</em>, not its volume —
        just like a hologram stores 3D information on a 2D surface. The bound is <strong className="text-purple-300">S ≤ A/4Gℏ</strong> (Bekenstein-Hawking).
      </p>
      <p>
        Juan Maldacena's 1997 <strong className="text-indigo-300">AdS/CFT correspondence</strong> gives a precise realization:
        a (d+1)-dimensional gravity theory in Anti-de Sitter space is <em>exactly equivalent</em> to a d-dimensional
        Conformal Field Theory on its boundary — with no gravity. The bulk (including black holes) is holographically
        encoded on the boundary.
      </p>
      <p>
        The <strong className="text-indigo-300">Ryu-Takayanagi formula</strong> (2006) connects entanglement entropy of a boundary
        region A to the area of the minimal geodesic surface in the bulk homologous to A: S(A) = Area/4Gₙ.
        This implies that spacetime geometry itself <em>emerges from entanglement structure</em> of the boundary theory.
      </p>
    </div>
  ),
}

// ─── Main component ───────────────────────────────────────────────────────────

export const HologramExplainer: React.FC = () => {
  const [activeLevel, setActiveLevel] = useState<LevelId>('eli5')
  const allParams = useAllParams()
  const level = LEVELS.find((l) => l.id === activeLevel)!

  const renderViz = () => {
    switch (activeLevel) {
      case 'eli5':
        return <PondRipples params={allParams.eli5.params} onParamChange={allParams.eli5.update} />
      case 'highschool':
        return <LaserHologram params={allParams.highschool.params} onParamChange={allParams.highschool.update} />
      case 'caustics':
        return <CausticsViz params={allParams.caustics.params} onParamChange={allParams.caustics.update} />
      case 'undergrad':
        return <FourierViz params={allParams.undergrad.params} onParamChange={allParams.undergrad.update} />
      case 'grad':
        return <ZonePlateViz params={allParams.grad.params} onParamChange={allParams.grad.update} />
      case 'phd':
        return <AdSCFTViz params={allParams.phd.params} onParamChange={allParams.phd.update} />
    }
  }

  const currentIdx = LEVELS.findIndex((l) => l.id === activeLevel)

  return (
    <div className="flex flex-col h-full overflow-hidden bg-gray-950 text-gray-100">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex-shrink-0 border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm px-5 py-3">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h1 className="font-mono font-bold text-lg text-sky-400 tracking-wide">
              Holography & the Holographic Principle
            </h1>
            <p className="text-xs font-mono text-gray-500 mt-0.5">
              From pond ripples to quantum gravity — interactive physics across 6 levels
            </p>
          </div>
          <div className="text-right">
            <div className={`font-mono text-sm font-bold ${level.tagText}`}>{level.label}</div>
            <div className="text-xs text-gray-500 font-mono">{level.sublabel}</div>
          </div>
        </div>

        {/* Level selector */}
        <div className="flex gap-1.5 items-stretch">
          {LEVELS.map((l, i) => {
            const isActive = l.id === activeLevel
            const isPast = i < currentIdx
            return (
              <button
                key={l.id}
                onClick={() => setActiveLevel(l.id)}
                className={`
                  flex-1 relative rounded-lg px-2 py-1.5 text-xs font-mono font-semibold
                  border transition-all duration-150 cursor-pointer
                  ${isActive
                    ? `bg-gradient-to-br ${l.gradient} text-white border-transparent shadow-lg`
                    : isPast
                      ? `${l.tagBg} ${l.tagText} ${l.borderColor} opacity-70 hover:opacity-100`
                      : `bg-gray-900/50 text-gray-500 border-gray-800 hover:border-gray-700 hover:text-gray-400`
                  }
                `}
              >
                <div className="text-center leading-tight">
                  <div>{l.label}</div>
                </div>
                {isActive && (
                  <div className={`absolute -bottom-px left-0 right-0 h-0.5 bg-gradient-to-r ${l.gradient} rounded-b`} />
                )}
              </button>
            )
          })}
        </div>

        {/* Complexity progress bar */}
        <div className="mt-2 flex items-center gap-2">
          <span className="text-xs font-mono text-gray-600">complexity</span>
          <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-sky-500 via-violet-500 to-indigo-500 transition-all duration-300"
              style={{ width: `${((currentIdx + 1) / LEVELS.length) * 100}%` }}
            />
          </div>
          <span className="text-xs font-mono text-gray-600">{currentIdx + 1}/{LEVELS.length}</span>
        </div>
      </div>

      {/* ── Content area ────────────────────────────────────────────────── */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-5xl mx-auto p-4 flex flex-col gap-4">
          {/* Level title and prose */}
          <div className={`rounded-xl border ${level.borderColor} ${level.bgColor} p-4`}>
            <h2 className={`font-mono font-bold text-base ${level.tagText} mb-3`}>
              {level.title}
            </h2>
            {LEVEL_PROSE[activeLevel]}
          </div>

          {/* Interactive visualization */}
          <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono text-gray-500 uppercase tracking-widest">
                Interactive Simulation
              </span>
              <span className={`text-xs font-mono px-2 py-0.5 rounded ${level.tagBg} ${level.tagText} border ${level.borderColor}`}>
                drag sliders to explore
              </span>
            </div>
            {renderViz()}
          </div>

          {/* Navigation buttons */}
          <div className="flex justify-between items-center pb-2">
            <button
              onClick={() => {
                if (currentIdx > 0) setActiveLevel(LEVELS[currentIdx - 1].id)
              }}
              disabled={currentIdx === 0}
              className={`
                font-mono text-xs px-4 py-2 rounded-lg border transition-all
                ${currentIdx === 0
                  ? 'border-gray-800 text-gray-700 cursor-not-allowed'
                  : 'border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300 cursor-pointer'}
              `}
            >
              ← {currentIdx > 0 ? LEVELS[currentIdx - 1].label : 'Start'}
            </button>

            <div className="flex gap-1">
              {LEVELS.map((l, i) => (
                <div
                  key={l.id}
                  className={`w-2 h-2 rounded-full cursor-pointer transition-all ${
                    i === currentIdx
                      ? `bg-gradient-to-r ${level.gradient}`
                      : i < currentIdx ? 'bg-gray-600' : 'bg-gray-800'
                  }`}
                  onClick={() => setActiveLevel(l.id)}
                />
              ))}
            </div>

            <button
              onClick={() => {
                if (currentIdx < LEVELS.length - 1) setActiveLevel(LEVELS[currentIdx + 1].id)
              }}
              disabled={currentIdx === LEVELS.length - 1}
              className={`
                font-mono text-xs px-4 py-2 rounded-lg border transition-all
                ${currentIdx === LEVELS.length - 1
                  ? 'border-gray-800 text-gray-700 cursor-not-allowed'
                  : 'border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300 cursor-pointer'}
              `}
            >
              {currentIdx < LEVELS.length - 1 ? LEVELS[currentIdx + 1].label : 'End'} →
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
