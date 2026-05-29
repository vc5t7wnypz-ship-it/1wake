import React, { useEffect, useRef, useCallback } from 'react'
import * as d3 from 'd3'
import { usePassageStore } from '../../store/passageStore'
import type { TokenPoint } from '../../store/passageStore'

const MARGIN = { top: 20, right: 120, bottom: 60, left: 50 }
const MAX_ENTROPY = 12 // ln(vocab_size) ≈ 12 nats for typical LLMs

interface LensData {
  name: string
  tokens: TokenPoint[]
  color: string
}

export const EntropyWaveform: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const { result, selectedPosition, setSelectedPosition } = usePassageStore()

  const draw = useCallback(() => {
    const container = containerRef.current
    const svg = svgRef.current
    if (!svg || !container) return

    const width = container.clientWidth
    const height = container.clientHeight
    if (width < 10 || height < 10) return

    d3.select(svg).selectAll('*').remove()

    const innerW = width - MARGIN.left - MARGIN.right
    const innerH = height - MARGIN.top - MARGIN.bottom

    const root = d3
      .select(svg)
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`)

    if (!result) {
      root
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#4b5563')
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('font-size', 12)
        .text('Submit a passage to view entropy waveform')
      return
    }

    const color = d3.scaleOrdinal(d3.schemeTableau10)
    const lensData: LensData[] = Object.entries(result.tokensByLens).map(
      ([name, tokens], i) => ({ name, tokens, color: color(String(i)) }),
    )

    if (lensData.length === 0) return

    const allTokens = lensData[0].tokens
    const positions = allTokens.map((t) => t.position)
    const tokenStrings = allTokens.map((t) => t.token)

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain([0, Math.max(1, positions.length - 1)])
      .range([0, innerW])

    const yScale = d3
      .scaleLinear()
      .domain([0, MAX_ENTROPY])
      .range([innerH, 0])
      .nice()

    // Grid lines
    const gridG = root.append('g').attr('class', 'grid')
    yScale.ticks(5).forEach((tick) => {
      gridG
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(tick))
        .attr('y2', yScale(tick))
        .attr('stroke', '#1f2937')
        .attr('stroke-width', 1)
    })

    // Superposition vertical markers
    result.superpositionPositions.forEach((pos) => {
      const idx = positions.indexOf(pos)
      if (idx < 0) return
      root
        .append('line')
        .attr('x1', xScale(idx))
        .attr('x2', xScale(idx))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#f59e0b')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,3')
        .attr('opacity', 0.6)
    })

    // Selected position highlight
    if (selectedPosition !== null) {
      const idx = positions.indexOf(selectedPosition)
      if (idx >= 0) {
        root
          .append('rect')
          .attr('x', xScale(idx) - 12)
          .attr('y', 0)
          .attr('width', 24)
          .attr('height', innerH)
          .attr('fill', '#0ea5e9')
          .attr('opacity', 0.12)
          .attr('rx', 2)
      }
    }

    // Lines + points per lens
    const lineGen = d3
      .line<TokenPoint>()
      .x((_, i) => xScale(i))
      .y((d) => yScale(Math.min(d.entropy, MAX_ENTROPY)))
      .curve(d3.curveCatmullRom.alpha(0.5))

    lensData.forEach((ld) => {
      // Line
      root
        .append('path')
        .datum(ld.tokens)
        .attr('fill', 'none')
        .attr('stroke', ld.color)
        .attr('stroke-width', 1.8)
        .attr('opacity', 0.85)
        .attr('d', lineGen)

      // Points
      const dotsG = root.append('g')

      ld.tokens.forEach((tok, i) => {
        const isSuperposition = result.superpositionPositions.includes(
          tok.position,
        )
        const cx = xScale(i)
        const cy = yScale(Math.min(tok.entropy, MAX_ENTROPY))

        if (isSuperposition) {
          // Diamond marker for superposition
          const r = 6
          dotsG
            .append('path')
            .attr(
              'd',
              `M${cx},${cy - r} L${cx + r},${cy} L${cx},${cy + r} L${cx - r},${cy} Z`,
            )
            .attr('fill', ld.color)
            .attr('stroke', '#f59e0b')
            .attr('stroke-width', 1.5)
            .style('cursor', 'pointer')
            .on('mousemove', (event: MouseEvent) => showTooltip(event, tok, ld.name))
            .on('mouseleave', hideTooltip)
            .on('click', () => setSelectedPosition(tok.position))
        } else {
          dotsG
            .append('circle')
            .attr('cx', cx)
            .attr('cy', cy)
            .attr('r', 3)
            .attr('fill', ld.color)
            .attr('opacity', 0.7)
            .style('cursor', 'pointer')
            .on('mousemove', (event: MouseEvent) => showTooltip(event, tok, ld.name))
            .on('mouseleave', hideTooltip)
            .on('click', () => setSelectedPosition(tok.position))
        }
      })
    })

    // Axes
    const xAxis = d3
      .axisBottom(xScale)
      .tickValues(positions.map((_, i) => i))
      .tickFormat((d) => {
        const idx = Number(d)
        return tokenStrings[idx]?.slice(0, 10) ?? ''
      })

    root
      .append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(xAxis)
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .attr('text-anchor', 'end')
      .attr('dx', '-0.4em')
      .attr('dy', '0.1em')
      .attr('fill', '#9ca3af')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-size', 9)

    root
      .selectAll('.domain, .tick line')
      .attr('stroke', '#374151')

    root
      .append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .attr('fill', '#9ca3af')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-size', 10)

    root
      .append('text')
      .attr('x', -innerH / 2)
      .attr('y', -38)
      .attr('transform', 'rotate(-90)')
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-size', 9)
      .text('entropy (nats)')

    // Legend
    const legendG = root
      .append('g')
      .attr('transform', `translate(${innerW + 8}, 0)`)

    lensData.forEach((ld, i) => {
      const yOff = i * 18
      legendG
        .append('line')
        .attr('x1', 0)
        .attr('x2', 14)
        .attr('y1', yOff + 6)
        .attr('y2', yOff + 6)
        .attr('stroke', ld.color)
        .attr('stroke-width', 2)
      legendG
        .append('text')
        .attr('x', 18)
        .attr('y', yOff + 10)
        .attr('fill', '#d1d5db')
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('font-size', 9)
        .text(ld.name.slice(0, 12))
    })
  }, [result, selectedPosition, setSelectedPosition])

  // Tooltip helpers
  const showTooltip = (event: MouseEvent, tok: TokenPoint, lens: string) => {
    const tt = tooltipRef.current
    if (!tt) return
    tt.innerHTML = `
      <div class="tt-label">${tok.token}</div>
      <div>lens: ${lens}</div>
      <div>entropy: ${tok.entropy.toFixed(3)} nats</div>
      <div>superposition: ${tok.superpositionScore.toFixed(3)}</div>
      <div>top field: ${tok.topField}</div>
    `
    tt.style.display = 'block'
    tt.style.left = `${event.pageX + 12}px`
    tt.style.top = `${event.pageY - 10}px`
  }

  const hideTooltip = () => {
    if (tooltipRef.current) tooltipRef.current.style.display = 'none'
  }

  useEffect(() => {
    draw()
  }, [draw])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => draw())
    ro.observe(container)
    return () => ro.disconnect()
  }, [draw])

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <svg ref={svgRef} className="w-full h-full" />
      <div
        ref={tooltipRef}
        className="d3-tooltip"
        style={{ display: 'none', position: 'fixed' }}
      />
    </div>
  )
}
