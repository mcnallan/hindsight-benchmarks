import { readFileSync } from 'node:fs'
import path from 'node:path'
import type { Board, CategoryResult, Row } from './data'
import STRUCTURE from './eval-structure.json'

const inklingSource = {
  params: '26.4B total, 3.4B active (5.4B effective with loops)',
  model_card: 'https://thinkingmachines.ai/model-card/inkling-2-mini/',
}

export const localProfiles = [
  { id: 'qwen35-4b-off', label: 'Qwen3.5 4B · Thinking off', icon: '/logos/qwen.webp', source: { params: '4B' } },
  { id: 'inkling-2-mini-off', label: 'Inkling 2 Mini · Thinking off', source: inklingSource },
  { id: 'inkling-2-mini-minimal', label: 'Inkling 2 Mini · Minimal effort', source: inklingSource },
]

type Operation = {
  passed: number
  total: number
  score: number
  errors: number
  timestamp: string
  categories?: Record<string, CategoryResult>
  tokens: { calls: number; input_tokens: number; output_tokens: number; thoughts_tokens?: number }
}

type LocalResult = {
  hindsight_commit: string
  judge: string
  full: boolean
  repeats: number
  operations: Record<string, Operation>
}

export const resultsRoot = path.resolve(process.cwd(), '..', 'results')

export function readOptional(file: string): string | undefined {
  try {
    return readFileSync(file, 'utf-8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
    throw error
  }
}

export function readResult<T>(file: string): T | undefined {
  const contents = readOptional(file)
  return contents ? (JSON.parse(contents) as T) : undefined
}

export function readLocalResult(id: string): LocalResult | undefined {
  return readResult<LocalResult>(path.join(resultsRoot, 'leaderboard', 'evals', `local-${id}-sol62-high.json`))
}

export function withLocalResults(board: Board): Board {
  if (board.group !== 'operation') return board
  const operation = board.id.replaceAll('-', '_')
  const caseCount = board.categories.reduce((sum, category) => sum + category.cases, 0)
  const localRows = localProfiles.flatMap((profile): Row[] => {
    const run = readLocalResult(profile.id)
    if (!run?.full || run.hindsight_commit !== STRUCTURE.hindsight_commit) return []
    const result = run.operations[operation]
    if (!result?.total || result.total !== caseCount * run.repeats) return []

    const { calls, input_tokens, output_tokens, thoughts_tokens = 0 } = result.tokens
    const perCase = (count: number) => Math.round(count / result.total)
    const day = result.timestamp.slice(0, 8)
    return [{
      id: `local-${profile.id}`,
      model: profile.label,
      provider: 'Local',
      icon: profile.icon,
      tag: 'local',
      quality: result.score,
      categories: result.categories ?? {},
      cases: { passed: result.passed, total: result.total },
      cost: null,
      callsPerTask: calls / result.total,
      tokensPerTask: { input: perCase(input_tokens), output: perCase(output_tokens + thoughts_tokens) },
      source: {
        ...profile.source,
        ran: `${day.slice(0, 4)}-${day.slice(4, 6)}-${day.slice(6, 8)}`,
        judge: run.judge,
        repeats: run.repeats,
      },
    }]
  })
  if (localRows.length === 0) return board
  const [main, ...otherTabs] = board.tabs
  return { ...board, tabs: [{ ...main, rows: [...main.rows, ...localRows] }, ...otherTabs] }
}
