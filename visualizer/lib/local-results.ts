import { readFileSync } from 'node:fs'
import path from 'node:path'
import { addScores, summarizeOperations, tokenTotals, type Board, type CategoryResult, type OpResult, type Row, type Source } from './data'
import STRUCTURE from './eval-structure.json'

const inklingSource = {
  params: '26.4B total, 3.4B active (5.4B effective with loops) · 4-bit expert quant',
  model_card: 'https://thinkingmachines.ai/model-card/inkling-2-mini/',
  price_per_1m: { input: 0.075, output: 0.30, cached_input: 0.015 },
}

type ProfileSource = Pick<Source, 'params' | 'model_card' | 'price_per_1m'>

export const localProfiles: { id: string; model: string; reasoningMode: string; icon?: string; source: ProfileSource }[] = [
  { id: 'qwen35-4b-off', model: 'Qwen3.5 4B', reasoningMode: 'Thinking disabled', icon: '/logos/qwen.webp', source: { params: '4B · 4-bit quant' } },
  { id: 'inkling-2-mini-off', model: 'Inkling 2 Mini', reasoningMode: 'Thinking disabled', source: inklingSource },
  { id: 'inkling-2-mini-minimal', model: 'Inkling 2 Mini', reasoningMode: 'Minimal effort', source: inklingSource },
]

type Operation = {
  passed: number
  total: number
  score: number
  errors: number
  timestamp: string
  categories?: Record<string, CategoryResult>
  tokens: { calls: number; input_tokens: number; cached_tokens?: number; output_tokens: number; thoughts_tokens?: number }
}

type LocalResult = {
  hindsight_commit: string
  judge: string
  full: boolean
  repeats: number
  timestamp: string
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

function operationResult(result: Operation | undefined, total: number, rates: Source['price_per_1m']) {
  if (!result?.total || result.total !== total) return

  const { calls, input_tokens, cached_tokens = 0 } = result.tokens
  const reasoning_tokens = result.tokens.thoughts_tokens
  const output_tokens = result.tokens.output_tokens + (reasoning_tokens ?? 0)
  return {
    passed: result.passed,
    total,
    score: result.score,
    timestamp: result.timestamp,
    cost_usd: rates
      ? ((input_tokens - cached_tokens) * rates.input + cached_tokens * rates.cached_input + output_tokens * rates.output) / 1_000_000
      : null,
    calls,
    input_tokens,
    cached_tokens: result.tokens.cached_tokens,
    output_tokens,
    reasoning_tokens,
    decision_calls: 0,
    decision_input_tokens: 0,
    decision_cost_usd: null,
    categories: result.categories ?? {},
  } satisfies OpResult & Pick<Operation, 'score' | 'timestamp'>
}

export function withLocalResults(board: Board): Board {
  const required = board.group === 'summary' ? board.categories : [{
    id: board.id.replaceAll('-', '_'),
    cases: board.categories.reduce((sum, category) => sum + category.cases, 0),
  }]
  const localRows = localProfiles.flatMap((profile): Row[] => {
    const run = readLocalResult(profile.id)
    if (!run?.full || run.hindsight_commit !== STRUCTURE.hindsight_commit) return []
    const operations = Object.fromEntries(required.map(({ id, cases }) => [
      id, operationResult(run.operations[id], cases * run.repeats, profile.source.price_per_1m),
    ]))
    const first = operations[required[0].id]
    if (!first) return []
    const stats = board.group === 'summary' ? summarizeOperations(operations) : {
      quality: first.score,
      categories: first.categories,
      cases: { passed: first.passed, total: first.total },
      cost: first.cost_usd === null ? null : 1000 * first.cost_usd / first.total,
      callsPerTask: first.calls / first.total,
      tokensPerTask: { input: Math.round(first.input_tokens / first.total), output: Math.round(first.output_tokens / first.total) },
      tokenTotals: tokenTotals([first]),
    }
    if (!stats) return []
    const day = (board.group === 'summary' ? run.timestamp : first.timestamp).slice(0, 8)
    return [{
      id: `local-${profile.id}`,
      model: profile.model,
      label: `${profile.model} · ${profile.reasoningMode}`,
      provider: 'Local',
      reasoningMode: profile.reasoningMode,
      icon: profile.icon,
      tag: 'local',
      ...stats,
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
  return { ...board, tabs: [{ ...main, rows: addScores([...main.rows, ...localRows]) }, ...otherTabs] }
}
