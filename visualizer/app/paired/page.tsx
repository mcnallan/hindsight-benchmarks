import type { Metadata } from 'next'
import Link from 'next/link'
import { ModelCell } from '@/components/ModelCell'
import { PageHero } from '@/components/Page'
import { ArmLabel, Legend, Redline, RepeatMarks } from '@/components/Paired'
import { PairedNav } from '@/components/PairedNav'
import { ARMS, OPERATIONS, armRate, fixtures, pairedCases, pairedDelta, pairedRuns } from '@/lib/paired'

export const metadata: Metadata = {
  title: 'Paired cases',
  description: 'Stock Hindsight system evals beside a reworded copy and a copy with one fact changed.',
}

export default function PairedPage() {
  const runs = pairedRuns()
  return (
    <>
      <PageHero
        title="Paired cases"
        subtitle="Twelve stock Retain and Reflect evals, each beside a reworded copy and a copy with one fact changed. Same pinned Hindsight, retrieval, judge and request controls, each in a fresh bank."
      />
      <PairedNav />
      <main className="max-w-6xl mx-auto px-4 py-8 space-y-12">
        <section className="grid gap-3 sm:grid-cols-3" aria-label="Arms">
          {ARMS.map((a) => (
            <div key={a.id} className="rounded-xl border border-border p-4">
              <ArmLabel arm={a.id}>{a.label}</ArmLabel>
              <p className="text-sm text-muted-foreground mt-1">{a.blurb}</p>
            </div>
          ))}
        </section>

        <section className="space-y-3">
          <h2 className="font-heading font-bold text-xl">Pass rate by arm</h2>
          <div className="overflow-x-auto rounded-xl border border-border">
            <table className="w-full min-w-[640px] text-sm">
              <thead className="bg-secondary text-left">
                <tr>
                  <th className="px-4 py-3">Profile</th>
                  {ARMS.map((a) => (
                    <th key={a.id} className="px-4 py-3">
                      <ArmLabel arm={a.id}>{a.label}</ArmLabel>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {runs.map((run) => (
                  <tr key={run.key}>
                    <th scope="row" className="px-4 py-3 text-left font-medium">
                      <ModelCell row={{ model: run.profile.model, reasoningMode: run.judge ? `${run.profile.reasoningMode} · ${run.judge} judge` : run.profile.reasoningMode, icon: run.profile.icon, provider: 'Local' }} />
                    </th>
                    {ARMS.map((a) => {
                      const { passed, graded } = armRate(run, a.id)
                      const delta = a.id === 'stock' ? undefined : pairedDelta(run, a.id)
                      return (
                        <td key={a.id} className="px-4 py-3 tabular-nums">
                          {graded ? (
                            <>
                              {((100 * passed) / graded).toFixed(1)}% <span className="text-muted-foreground">{passed}/{graded}</span>
                              {delta && (
                                <span
                                  className={`ml-2 font-mono text-xs ${delta.pp < 0 ? 'text-[hsl(var(--bad))]' : 'text-muted-foreground'}`}
                                  title={`Against the matched stock execution, over ${delta.n} case repeats`}
                                >
                                  {delta.pp >= 0 ? '+' : '−'}
                                  {Math.abs(delta.pp).toFixed(1)} pp
                                </span>
                              )}
                            </>
                          ) : (
                            <span className="text-muted-foreground">Pending</span>
                          )}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted-foreground">
            Change is the variant&apos;s pass rate minus stock&apos;s on the same case and repeat. Infrastructure failures are left out of
            every rate.
          </p>
        </section>

        {OPERATIONS.map((op) => (
          <section key={op.id} className="space-y-3">
            <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
              <h2 className="font-heading font-bold text-xl">{op.label}</h2>
              <p className="text-sm text-muted-foreground">{op.about}</p>
            </div>
            <div className="overflow-x-auto rounded-xl border border-border">
              <table className="w-full min-w-[640px] text-sm">
                <thead className="bg-secondary text-left">
                  <tr>
                    <th className="px-4 py-3 w-[44%]">Case, and how the counterfactual changes its answer</th>
                    {runs.map((run) => (
                      <th key={run.key} className="px-4 py-3 font-normal">
                        <div className="font-medium">{run.profile.model}</div>
                        <div className="text-xs text-muted-foreground">{run.profile.reasoningMode}{run.judge && ` · ${run.judge} judge`}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {pairedCases
                    .filter((c) => c.operation === op.id)
                    .map((c) => {
                      const cf = c.arms.find((a) => a.arm === 'counterfactual')
                      const stockAnswer = c.arms.find((a) => a.arm === 'stock')?.answer ?? ''
                      // A short edit reads best inline; a rewritten sentence reads best as before and after.
                      const inline = (cf?.answer_diff?.filter((s) => s.op !== 'eq').length ?? 99) <= 4
                      return (
                        <tr key={c.id} className="align-top">
                          <td className="px-4 py-3">
                            <Link href={`/paired/${c.id}`} className="font-mono text-[13px] font-medium hover:text-[hsl(var(--primary))]">
                              {c.id}
                            </Link>
                            <span className="ml-2 text-xs text-muted-foreground">{c.category.replace('_', ' ')}</span>
                            {cf?.answer_diff &&
                              (inline ? (
                                <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">
                                  Answer: <Redline segments={cf.answer_diff} arm="counterfactual" />
                                </p>
                              ) : (
                                <div className="mt-1 text-[13px] leading-relaxed text-muted-foreground space-y-0.5">
                                  <p><del className="rl-del">{stockAnswer}</del></p>
                                  <p><ins className="rl-ins-cf">{cf.answer}</ins></p>
                                </div>
                              ))}
                          </td>
                          {runs.map((run) => (
                            <td key={run.key} className="px-4 py-3">
                              <div className="grid grid-cols-[auto_auto] gap-x-2 gap-y-1 items-center w-max">
                                {ARMS.map((a) => (
                                  <span key={a.id} className="contents">
                                    <span className={`text-[11px] font-heading font-bold ${a.id === 'stock' ? 'text-muted-foreground' : a.id === 'wording' ? 'text-[hsl(var(--primary))]' : 'text-[hsl(var(--cf))]'}`}>
                                      {a.label[0]}
                                    </span>
                                    <RepeatMarks runs={run.byCase[c.id]?.[a.id]} repeats={run.report?.repeats ?? 2} label={`${c.id} ${a.label}`} />
                                  </span>
                                ))}
                              </div>
                            </td>
                          ))}
                        </tr>
                      )
                    })}
                </tbody>
              </table>
            </div>
          </section>
        ))}

        <div className="space-y-2">
          <Legend />
          <p className="text-xs text-muted-foreground">
            Fixture revision <span className="font-mono">{fixtures.fixture_revision}</span> on Hindsight{' '}
            <span className="font-mono">{fixtures.pinned_commit.slice(0, 8)}</span>. Each row names the codex-lb judge that graded it;
            compare profiles only across rows with the same judge.
          </p>
        </div>
      </main>
    </>
  )
}
