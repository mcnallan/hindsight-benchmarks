import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import Link from 'next/link'
import { PageHero } from '@/components/Page'

type Result = {
  model_id: string
  quality?: {
    dataset?: string
    model_label?: string
    partial?: boolean
    correct: number
    total: number
    retrieval_profile?: string
    judge_model?: string
  }
}

const resultDir = path.join(process.cwd(), '..', 'results', 'leaderboard', 'llm')

export default function HistoricalRetainPage() {
  const rows = readdirSync(resultDir)
    .filter((name) => name.endsWith('.json'))
    .flatMap((name) => {
      const { model_id, quality } = JSON.parse(readFileSync(path.join(resultDir, name), 'utf-8')) as Result
      if (quality?.dataset !== 'beam_128k_subset' || quality.partial || !quality.total) return []
      return [{ id: name, model: quality.model_label ?? model_id, ...quality }]
    })
    .sort((a, b) => b.correct / b.total - a.correct / a.total || a.model.localeCompare(b.model))

  return (
    <>
      <PageHero
        title="Historical BEAM retain results"
        subtitle="Completed results from the four-conversation, facts-only BEAM benchmark. Each row shows the retrieval and judging setup recorded by its run."
      />
      <main className="max-w-6xl mx-auto px-4 py-8 space-y-4">
        <Link className="text-[hsl(var(--primary))] underline" href="/leaderboards/retain">
          Hindsight system retain evals
        </Link>
        <div className="overflow-x-auto rounded-xl border border-border">
          <table className="w-full text-sm">
            <thead className="bg-secondary text-left">
              <tr>
                {['Retain model', 'Correct', 'Accuracy', 'Retrieval', 'Judge'].map((title) => (
                  <th key={title} className="px-4 py-3">{title}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((row) => (
                <tr key={row.id}>
                  <td className="px-4 py-3">{row.model}</td>
                  <td className="px-4 py-3 tabular-nums whitespace-nowrap">{row.correct}/{row.total}</td>
                  <td className="px-4 py-3 tabular-nums">{(100 * row.correct / row.total).toFixed(2)}%</td>
                  <td className="px-4 py-3">{row.retrieval_profile ?? 'Unrecorded'}</td>
                  <td className="px-4 py-3">{row.judge_model ?? 'Unrecorded'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </>
  )
}
