import React, {
  useEffect,
  useRef,
  useCallback,
  useState,
} from 'react'
import * as d3 from 'd3'
import { usePassageStore } from '../../store/passageStore'
import { getGraphNode } from '../../api/client'

// ── Types ────────────────────────────────────────────────────────────────────

type NodeType =
  | 'WakeToken'
  | 'Root'
  | 'SemanticField'
  | 'MythFigure'
  | 'KabbalahNode'
  | string

interface GraphNode extends d3.SimulationNodeDatum {
  id: string
  type: NodeType
  label: string
  meaning?: string
  expanded?: boolean
}

interface GraphEdge extends d3.SimulationLinkDatum<GraphNode> {
  relationship: string
  weight: number
}

const NODE_COLOR: Record<string, string> = {
  WakeToken: '#3b82f6',     // blue
  Root: '#22c55e',          // green
  SemanticField: '#f59e0b', // amber
  MythFigure: '#a855f7',    // purple
  KabbalahNode: '#eab308',  // gold
}

const ALL_TYPES = Object.keys(NODE_COLOR) as NodeType[]

// ── Component ────────────────────────────────────────────────────────────────

export const GraphExplorer: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphEdge> | null>(null)

  const { result } = usePassageStore()
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [visibleTypes, setVisibleTypes] = useState<Set<NodeType>>(
    new Set(ALL_TYPES),
  )
  const [noData, setNoData] = useState(false)

  // Build initial graph from result
  useEffect(() => {
    if (!result) {
      setNodes([])
      setEdges([])
      setNoData(false)
      return
    }

    const allTokens = Object.values(result.tokensByLens)
      .flat()
      .filter(
        (t, idx, arr) => arr.findIndex((u) => u.token === t.token) === idx,
      )
      .slice(0, 20)

    if (allTokens.length === 0) {
      setNoData(true)
      return
    }

    setNoData(false)
    const newNodes: GraphNode[] = allTokens.map((t) => ({
      id: t.token,
      type: 'WakeToken',
      label: t.token,
      meaning: `top field: ${t.topField}`,
    }))

    // Add semantic field nodes from topField values
    const fields = [...new Set(allTokens.map((t) => t.topField))]
    fields.forEach((f) => {
      if (f && !newNodes.find((n) => n.id === f)) {
        newNodes.push({ id: f, type: 'SemanticField', label: f })
      }
    })

    const newEdges: GraphEdge[] = allTokens
      .filter((t) => t.topField)
      .map((t) => ({
        source: t.token,
        target: t.topField,
        relationship: 'activates',
        weight: t.superpositionScore,
      }))

    setNodes(newNodes)
    setEdges(newEdges)
  }, [result])

  const expandNode = useCallback(
    async (node: GraphNode) => {
      if (node.expanded) return
      try {
        const resp = await getGraphNode(node.id)
        const existingIds = new Set(nodes.map((n) => n.id))
        const newNodes = resp.neighbors
          .filter((nb) => !existingIds.has(nb.surface))
          .map(
            (nb): GraphNode => ({
              id: nb.surface,
              type: nb.type as NodeType,
              label: nb.surface,
            }),
          )
        const newEdges = resp.neighbors.map(
          (nb): GraphEdge => ({
            source: node.id,
            target: nb.surface,
            relationship: nb.relationship,
            weight: nb.weight,
          }),
        )
        setNodes((prev) => [
          ...prev.map((n) =>
            n.id === node.id ? { ...n, expanded: true } : n,
          ),
          ...newNodes,
        ])
        setEdges((prev) => [...prev, ...newEdges])
      } catch {
        // silently ignore graph expansion errors
      }
    },
    [nodes],
  )

  const draw = useCallback(() => {
    const container = containerRef.current
    const svg = svgRef.current
    if (!svg || !container) return

    const width = container.clientWidth
    const height = container.clientHeight
    if (width < 10 || height < 10) return

    d3.select(svg).selectAll('*').remove()
    if (simulationRef.current) simulationRef.current.stop()

    if (noData || nodes.length === 0) {
      d3.select(svg)
        .attr('width', width)
        .attr('height', height)
        .append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#4b5563')
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('font-size', 11)
        .text('No graph data — connect Neo4j to enable')
      return
    }

    const visNodes = nodes.filter((n) => visibleTypes.has(n.type))
    const visEdges = edges.filter((e) => {
      const srcId = typeof e.source === 'string' ? e.source : (e.source as GraphNode).id
      const tgtId = typeof e.target === 'string' ? e.target : (e.target as GraphNode).id
      return (
        visNodes.find((n) => n.id === srcId) &&
        visNodes.find((n) => n.id === tgtId)
      )
    })

    const svgEl = d3
      .select(svg)
      .attr('width', width)
      .attr('height', height)

    // Zoom group
    const g = svgEl.append('g')
    svgEl.call(
      d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.2, 4]).on('zoom', (e) => {
        g.attr('transform', e.transform)
      }),
    )

    // Arrow marker
    svgEl
      .append('defs')
      .append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', 14)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L8,0L0,4')
      .attr('fill', '#374151')

    // Edges
    const link = g
      .append('g')
      .selectAll<SVGLineElement, GraphEdge>('line')
      .data(visEdges)
      .join('line')
      .attr('stroke', '#374151')
      .attr('stroke-width', (d) => Math.max(0.5, d.weight * 3))
      .attr('marker-end', 'url(#arrow)')
      .attr('opacity', 0.6)

    // Nodes group
    const node = g
      .append('g')
      .selectAll<SVGGElement, GraphNode>('g')
      .data(visNodes, (d) => d.id)
      .join('g')
      .style('cursor', 'pointer')
      .call(
        d3
          .drag<SVGGElement, GraphNode>()
          .on('start', (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart()
            d.fx = d.x
            d.fy = d.y
          })
          .on('drag', (event, d) => {
            d.fx = event.x
            d.fy = event.y
          })
          .on('end', (event, d) => {
            if (!event.active) sim.alphaTarget(0)
            d.fx = null
            d.fy = null
          }),
      )
      .on('click', (_event, d) => expandNode(d))
      .on('mousemove', (event: MouseEvent, d) => {
        const tt = tooltipRef.current
        if (!tt) return
        tt.innerHTML = `
          <div class="tt-label">${d.label}</div>
          <div>type: ${d.type}</div>
          ${d.meaning ? `<div>meaning: ${d.meaning}</div>` : ''}
          ${d.expanded ? '' : '<div style="color:#9ca3af;margin-top:4px">click to expand</div>'}
        `
        tt.style.display = 'block'
        tt.style.left = `${event.pageX + 12}px`
        tt.style.top = `${event.pageY - 10}px`
      })
      .on('mouseleave', () => {
        if (tooltipRef.current) tooltipRef.current.style.display = 'none'
      })

    node
      .append('circle')
      .attr('r', (d) => (d.type === 'WakeToken' ? 8 : 6))
      .attr('fill', (d) => NODE_COLOR[d.type] ?? '#6b7280')
      .attr('stroke', (d) => (d.expanded ? '#e5e7eb' : 'transparent'))
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.85)

    node
      .append('text')
      .attr('dy', 16)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-size', 8)
      .text((d) => d.label.slice(0, 12))

    // Simulation
    const sim = d3
      .forceSimulation<GraphNode>(visNodes)
      .force(
        'link',
        d3
          .forceLink<GraphNode, GraphEdge>(visEdges)
          .id((d) => d.id)
          .distance(70),
      )
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(18))

    simulationRef.current = sim

    sim.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as GraphNode).x ?? 0)
        .attr('y1', (d) => (d.source as GraphNode).y ?? 0)
        .attr('x2', (d) => (d.target as GraphNode).x ?? 0)
        .attr('y2', (d) => (d.target as GraphNode).y ?? 0)

      node.attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`)
    })
  }, [nodes, edges, visibleTypes, noData, expandNode])

  useEffect(() => {
    draw()
    return () => { simulationRef.current?.stop() }
  }, [draw])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => draw())
    ro.observe(container)
    return () => ro.disconnect()
  }, [draw])

  const toggleType = (type: NodeType) => {
    setVisibleTypes((prev) => {
      const next = new Set(prev)
      if (next.has(type)) {
        if (next.size > 1) next.delete(type)
      } else {
        next.add(type)
      }
      return next
    })
  }

  return (
    <div className="w-full h-full flex flex-col">
      {/* Filter bar */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-900/50 flex-wrap border-b border-gray-800/60">
        {ALL_TYPES.map((type) => {
          const active = visibleTypes.has(type)
          const color = NODE_COLOR[type]
          return (
            <button
              key={type}
              onClick={() => toggleType(type)}
              className={`
                flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono
                border transition-all duration-150
                ${active ? 'border-current opacity-90' : 'border-gray-700 opacity-30'}
              `}
              style={active ? { color, borderColor: color + '80' } : undefined}
            >
              <span
                className="w-2 h-2 rounded-full"
                style={{ background: active ? color : '#374151' }}
              />
              {type}
            </button>
          )
        })}
      </div>
      <div ref={containerRef} className="flex-1 relative">
        <svg ref={svgRef} className="w-full h-full" />
        <div
          ref={tooltipRef}
          className="d3-tooltip"
          style={{ display: 'none', position: 'fixed' }}
        />
      </div>
    </div>
  )
}
