import { readdirSync } from 'node:fs'
import path from 'node:path'
import { localProfiles, readResult, resultsRoot } from './local-results'
import FIXTURES from './paired-cases.json'

// Exported by benchmark-runner: python -m hindsight_benchmark.paired.export_visualizer lib/paired-cases.json
export type Segment = { op: 'eq' | 'del' | 'ins'; text: string }
export type Item = { text: string; tag?: string; diff?: Segment[] }
export type Section = { label: string; hint: string; items: Item[] }
export type ArmName = 'stock' | 'wording' | 'counterfactual'
export type Arm = {
  arm: ArmName
  change: string
  answer: string
  answer_diff: Segment[] | null
  notes: string
  sections: Section[]
  probes: { label: string; response: string; expect_pass: boolean; expect_stale: boolean | null }[]
}
export type PairedCase = {
  id: string
  operation: 'retain' | 'reflect'
  kind: string
  category: string
  origin: string
  prompt: string
  arms: Arm[]
}

export const fixtures = FIXTURES as { fixture_set: string; fixture_revision: string; pinned_commit: string; generated: string; cases: PairedCase[] }
export const pairedCases = fixtures.cases
export const ARMS: { id: ArmName; label: string; blurb: string }[] = [
  { id: 'stock', label: 'Stock', blurb: 'The upstream case as pinned' },
  { id: 'wording', label: 'Wording', blurb: 'New names and phrasing, same answer' },
  { id: 'counterfactual', label: 'Counterfactual', blurb: 'One fact changed, so the answer changes' },
]
export const OPERATIONS = [
  { id: 'retain', label: 'Retain', about: 'Graded on the facts the bank stores from one input.' },
  { id: 'reflect', label: 'Reflect', about: 'Graded on the answer to one question over a seeded bank.' },
] as const

export type Execution = {
  repeat: number
  case_id: string
  arm: ArmName
  passed: boolean | null
  hit_trap: boolean | null
  stale: boolean | null
  blame: string
  gold_retrieved: number | null
  gold_total: number | null
  infra_error?: string
}
type Report = {
  result_id: string
  fixture_set: string
  fixture_revision: string
  repeats: number
  settings: { case_ids: string[]; suite: string; judge_model: string; judge_reasoning_effort: string }
  executions: Execution[]
}

export type ProfileRun = {
  key: string
  profile: (typeof localProfiles)[number]
  /** The grader, such as "GPT-6.1 Terra medium"; empty while a profile has no report. */
  judge: string
  report?: Report
  byCase: Record<string, Partial<Record<ArmName, Execution[]>>>
}

// "gpt-6.1-terra", "medium" -> "GPT-6.1 Terra medium".
function judgeLabel(model: string, effort: string) {
  const [, version, name] = /^gpt-([\d.]+)-(\w+)$/.exec(model) ?? [, model, '']
  return `GPT-${version}${name ? ` ${name[0].toUpperCase()}${name.slice(1)}` : ''} ${effort}`
}

// The full paired job for each local profile and judge on this fixture set: every case, the most
// repeats. A profile graded by two judges gets two rows; one with no report yet shows as pending.
export function pairedRuns(): ProfileRun[] {
  const dir = path.join(resultsRoot, 'paired')
  let files: string[] = []
  try {
    files = readdirSync(dir).filter((f) => f.endsWith('.json'))
  } catch {}
  const reports = files
    .map((f) => readResult<Report>(path.join(dir, f)))
    .filter((r): r is Report => !!r && r.fixture_set === fixtures.fixture_set && r.settings.suite === 'paired')
    .filter((r) => r.settings.case_ids.length === 0)
  return localProfiles.flatMap((profile): ProfileRun[] => {
    const mine = reports.filter((r) => r.result_id.startsWith(`local-${profile.id}-`))
    const judges = [...new Set(mine.map((r) => judgeLabel(r.settings.judge_model, r.settings.judge_reasoning_effort)))].sort()
    if (!judges.length) return [{ key: profile.id, profile, judge: '', byCase: {} }]
    return judges.map((judge) => {
      const report = mine
        .filter((r) => judgeLabel(r.settings.judge_model, r.settings.judge_reasoning_effort) === judge)
        .sort((a, b) => b.repeats - a.repeats || b.executions.length - a.executions.length)[0]
      const byCase: ProfileRun['byCase'] = {}
      for (const e of report.executions) {
        ;((byCase[e.case_id] ??= {})[e.arm] ??= []).push(e)
      }
      return { key: `${profile.id}:${judge}`, profile, judge, report, byCase }
    })
  })
}

export function armRate(run: ProfileRun, arm: ArmName) {
  const rows = (run.report?.executions ?? []).filter((e) => e.arm === arm && e.passed !== null)
  return { passed: rows.filter((e) => e.passed).length, graded: rows.length }
}

// Variant minus stock over matched (case, repeat) pairs, both graded.
export function pairedDelta(run: ProfileRun, arm: ArmName) {
  let diff = 0
  let n = 0
  for (const arms of Object.values(run.byCase)) {
    for (const v of arms[arm] ?? []) {
      const s = arms.stock?.find((x) => x.repeat === v.repeat)
      if (!s || s.passed === null || v.passed === null) continue
      diff += Number(v.passed) - Number(s.passed)
      n++
    }
  }
  return n ? { pp: (100 * diff) / n, n } : undefined
}
