import path from 'node:path'
import { ModelCell, tokenTip } from '@/components/ModelCell'
import { BoardNav, PageHero } from '@/components/Page'
import { localProfiles as profiles, readLocalResult, readOptional, readResult, resultsRoot } from '@/lib/local-results'

const operations = [
  ['retain', 'Retain'],
  ['reflect', 'Reflect'],
  ['consolidation', 'Consolidation'],
  ['refresh_pages', 'Refresh pages'],
]

const benchmarks = [
  ['precisionmembench-single-turn', 'PrecisionMemBench'],
  ['locomo-fixed-locomo10', 'LoCoMo Fixed · Reflect'],
  ['locomo-locomo10', 'LoCoMo · Recall and answer'],
  ['beam-100k', 'BEAM 100K'],
  ['personamem-32k', 'PersonaMem 32K'],
  ['lifebench-en', 'LifeBench EN'],
  ['longmemeval-s', 'LongMemEval S'],
]

type Benchmark = { correct: number; total_queries: number; accuracy: number }
type Cell = { correct: number; total: number; percent: number; errors?: number; tokens?: string }

export default function LocalLeaderboardsPage() {
  const runsRoot = path.join(resultsRoot, 'runs')
  const active = readOptional(path.join(runsRoot, 'current-system-evals'))?.trim()
  const runDir = active?.startsWith(`${runsRoot}${path.sep}`) ? active : undefined
  const system = profiles.map(({ id }) => readLocalResult(id)?.operations)
  const systemRows = operations.map(([operation, label]) => ({
    label,
    cells: system.map((profile) => {
      const result = profile?.[operation]
      if (!result) return undefined
      const { calls, input_tokens, cached_tokens, output_tokens, thoughts_tokens } = result.tokens
      const tokens = tokenTip({
        cases: result.total,
        calls,
        input: input_tokens,
        cached: cached_tokens,
        output: output_tokens + (thoughts_tokens ?? 0),
        reasoning: thoughts_tokens,
      })
      return { correct: result.passed, total: result.total, percent: result.score, errors: result.errors, tokens }
    }),
  }))
  const benchmarkRows = benchmarks.map(([benchmark, label]) => ({
    label,
    cells: profiles.map(({ id }) => {
      if (!runDir) return undefined
      const result = readResult<Benchmark>(path.join(runDir, id, 'amb', benchmark, 'summary.json'))
      return result && { correct: result.correct, total: result.total_queries, percent: result.accuracy * 100 }
    }),
  }))

  return (
    <>
      <PageHero title="Local model results" subtitle="Production Qwen and experimental Inkling on Hindsight 1.2.0 with BGE-small, MiniLM-L6, and GPT-6.2 Sol high grading." />
      <BoardNav current="local" />
      <main className="max-w-6xl mx-auto px-4 py-8 space-y-8">
        <ResultTable title="Hindsight operations" rows={systemRows} />
        <ResultTable title="Agent Memory Benchmark" rows={benchmarkRows} />
      </main>
    </>
  )
}

function ResultTable({ title, rows }: { title: string; rows: { label: string; cells: (Cell | undefined)[] }[] }) {
  return (
    <section className="space-y-3">
      <h2 className="font-heading font-bold text-xl">{title}</h2>
      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="w-full text-sm">
          <thead className="bg-secondary text-left">
            <tr>
              <th className="px-4 py-3">Suite</th>
              {profiles.map(({ id, model, reasoningMode, icon }) => (
                <th key={id} className="px-4 py-3">
                  <ModelCell row={{ model, reasoningMode, icon, provider: 'Local' }} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {rows.map(({ label, cells }) => (
              <tr key={label}>
                <th scope="row" className="px-4 py-3 text-left font-medium">{label}</th>
                {cells.map((cell, i) => (
                  <td key={profiles[i].id} className={`px-4 py-3 tabular-nums ${cell?.tokens ? 'cursor-help' : ''}`} title={cell?.tokens}>
                    {cell ? (
                      <>
                        {cell.percent?.toFixed(1) ?? '—'}% <span className="text-muted-foreground">{cell.correct}/{cell.total}</span>
                        {!!cell.errors && <span className="text-muted-foreground"> · {cell.errors} errors</span>}
                      </>
                    ) : <span className="text-muted-foreground">Pending</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
