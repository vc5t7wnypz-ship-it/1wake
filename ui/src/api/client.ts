import axios from 'axios'
import type { ContrastiveResult } from '../store/passageStore'

const http = axios.create({
  baseURL: '/api/wake',
  timeout: 60_000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Request / Response types ────────────────────────────────────────────────

export interface PassageRequest {
  passage: string
  page: number
  line: number
  lenses: string[]
}

export interface GraphNodeResponse {
  surface: string
  type: string
  meaning?: string
  etymology?: string
  languages?: string[]
  neighbors: Array<{
    surface: string
    type: string
    relationship: string
    weight: number
  }>
}

export interface GraphPathResponse {
  from: string
  to: string
  path: string[]
  totalWeight: number
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  neo4j: boolean
  model: string
  version: string
}

// ── API functions ────────────────────────────────────────────────────────────

export async function analyzePassage(
  req: PassageRequest,
): Promise<ContrastiveResult> {
  const { data } = await http.post<ContrastiveResult>('/analyze', req)
  return data
}

export async function getGraphNode(
  surface: string,
): Promise<GraphNodeResponse> {
  const { data } = await http.get<GraphNodeResponse>(
    `/graph/node/${encodeURIComponent(surface)}`,
  )
  return data
}

export async function getGraphPath(
  from: string,
  to: string,
): Promise<GraphPathResponse> {
  const { data } = await http.get<GraphPathResponse>(
    `/graph/path/${encodeURIComponent(from)}/${encodeURIComponent(to)}`,
  )
  return data
}

export async function getLenses(): Promise<string[]> {
  const { data } = await http.get<string[]>('/lenses')
  return data
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await http.get<HealthResponse>('/health')
  return data
}
