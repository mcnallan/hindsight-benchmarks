import type { Metadata } from 'next'
import Link from 'next/link'
import { ChevronRight } from 'lucide-react'
import { notFound } from 'next/navigation'
import { CleanCopy } from '@/components/CleanCopy'
import { ArmLabel, ItemText, Legend, Redline, RepeatMarks } from '@/components/Paired'
import { PairedNav } from '@/components/PairedNav'
import { ARMS, OPERATIONS, pairedCases, pairedRuns, type Arm, type ArmName, type Execution, type Section } from '@/lib/paired'
import { cn } from '@/lib/utils'

export function generateStaticParams() {
  return pairedCases.map((c) => ({ case: c.id }))
}

export async function generateMetadata({ params }: { params: Promise<{ case: string }> }): Promise<Metadata> {
  const { case: id } = await params
  return { title: `${id} · Paired cases`, description: `Stock, wording and counterfactual versions of ${id}.` }
}

const BLAME: Record<string, string> = {
  extraction: 'extraction',
  retrieval: 'retrieval: a gold row never reached the model',
  generation: 'generation: every gold row arrived',
  deterministic: 'a deterministic check',
  infrastructure: 'infrastructure',
}

export default async function PairedCasePage({ params }: { params: Promise<{ case: string }> }) {
  const { case: id } = await params
  const c = pairedCases.find((x) => x.id === id)
  if (!c) notFound()
  const arms = ARMS.map((a) => c.arms.find((x) => x.arm === a.id)).filter((a): a is Arm => !!a)
  // Every section label any arm has, in first-seen order: a counterfactual can add a trap or a check.
  const labels = [...new Set(arms.flatMap((a) => a.sections.map((s) => s.label)))]
  const runs = pairedRuns()
  const op = OPERATIONS.find((o) => o.id === c.operation)!

  return (
    <>
      <div className="bg-[hsl(var(--band))] border-b border-border">
        <div className="max-w-6xl mx-auto px-4 pt-10 pb-8">
          <p className="text-sm text-muted-foreground flex items-center gap-1">
            <Link href="/paired" className="hover:text-foreground">Paired cases</Link>
            <ChevronRight className="w-3.5 h-3.5" aria-hidden />
            {op.label}
            <ChevronRight className="w-3.5 h-3.5" aria-hidden />
            {c.category.replace('_', ' ')}
          </p>
          <h1 className="font-mono text-[1.6rem] sm:text-[2.1rem] font-medium tracking-[-0.02em] mt-2">{c.id}</h1>
          {c.prompt && <p className="mt-3 text-lg font-heading font-bold max-w-3xl">“{c.prompt}”</p>}
          {c.origin && <p className="mt-2 text-sm text-muted-foreground max-w-3xl">Upstream origin: {c.origin}</p>}
        </div>
      </div>
      <PairedNav current={c.id} />

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-10">
        <section aria-label="Correct answer per arm" className="grid gap-px bg-border border border-border rounded-xl overflow-hidden lg:grid-cols-3">
          {arms.map((a) => (
            <div key={a.arm} className="bg-background p-4 sm:p-5">
              <ArmLabel arm={a.arm}>{ARMS.find((x) => x.id === a.arm)!.label}</ArmLabel>
              <div className="mt-2 text-xs text-muted-foreground">Correct answer</div>
              <div className="font-heading font-bold text-lg leading-snug">
                {a.answer_diff ? <Redline segments={a.answer_diff} arm={a.arm} /> : a.answer}
              </div>
              <p className="mt-3 text-sm text-muted-foreground leading-relaxed">{a.change || 'The upstream case, exactly as pinned.'}</p>
            </div>
          ))}
        </section>

        <section className="space-y-3">
          <h2 className="font-heading font-bold text-xl">Pilot results</h2>
          <div className="overflow-x-auto rounded-xl border border-border">
            <table className="w-full min-w-[640px] text-sm">
              <thead className="bg-secondary text-left">
                <tr>
                  <th className="px-4 py-3">Profile</th>
                  {arms.map((a) => (
                    <th key={a.arm} className="px-4 py-3">
                      <ArmLabel arm={a.arm}>{ARMS.find((x) => x.id === a.arm)!.label}</ArmLabel>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {runs.map((run) => (
                  <tr key={run.key} className="align-top">
                    <th scope="row" className="px-4 py-3 text-left font-normal">
                      <div className="font-medium">{run.profile.model}</div>
                      <div className="text-xs text-muted-foreground">{run.profile.reasoningMode}{run.judge && ` · ${run.judge} judge`}</div>
                    </th>
                    {arms.map((a) => (
                      <td key={a.arm} className="px-4 py-3">
                        <RepeatMarks runs={run.byCase[c.id]?.[a.arm]} repeats={run.report?.repeats ?? 2} label={`${run.profile.model} ${run.profile.reasoningMode} ${run.judge}, ${a.arm}`} />
                        <Outcomes runs={run.byCase[c.id]?.[a.arm]} pending={!run.report} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="space-y-3">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="font-heading font-bold text-xl">Side by side</h2>
              <p className="text-sm text-muted-foreground">Each variant is marked against stock. Switch to clean copy to read an arm as the model and judge see it.</p>
            </div>
          </div>
          <CleanCopy>
            <div className="hidden lg:grid grid-cols-3 gap-6 pb-2 border-b border-border">
              {arms.map((a) => (
                <ArmLabel key={a.arm} arm={a.arm}>{ARMS.find((x) => x.id === a.arm)!.label}</ArmLabel>
              ))}
            </div>
            <div className="divide-y divide-border">
              {labels.map((label) => {
                const cells = arms.map((a) => ({ arm: a.arm, section: a.sections.find((s) => s.label === label) }))
                const hint = cells.map((x) => x.section?.hint).find(Boolean)
                return (
                  <div key={label} className="py-5">
                    <h3 className="font-heading font-bold text-[15px]">{label}</h3>
                    {hint && <p className="text-xs text-muted-foreground mt-0.5">{hint}</p>}
                    <div className="mt-3 grid gap-4 lg:grid-cols-3 lg:gap-6">
                      {cells.map(({ arm, section }) => (
                        <Cell key={arm} arm={arm} section={section} />
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </CleanCopy>
        </section>

        <section className="space-y-3">
          <h2 className="font-heading font-bold text-xl">Grader checks</h2>
          <p className="text-sm text-muted-foreground max-w-3xl">
            Canned outputs graded with each arm&apos;s criteria and the run&apos;s judge before any model ran. Each variant must accept its own
            answer and reject the stock one.
          </p>
          <div className="grid gap-4 lg:grid-cols-3">
            {arms.map((a) => (
              <div key={a.arm} className="space-y-2">
                <ArmLabel arm={a.arm} className="lg:hidden">{ARMS.find((x) => x.id === a.arm)!.label}</ArmLabel>
                {a.probes.length ? (
                  a.probes.map((p) => (
                    <details key={p.label} className="group rounded-xl border border-border">
                      <summary className="cursor-pointer list-none px-3 py-2 text-sm flex gap-2 items-baseline focus-visible:outline focus-visible:outline-2 focus-visible:outline-[hsl(var(--ring))]">
                        <span className={cn('shrink-0 font-mono text-[11px] px-1.5', p.expect_pass ? 'text-[hsl(var(--good))]' : 'text-[hsl(var(--bad))]')}>
                          {p.expect_pass ? 'must pass' : 'must fail'}
                        </span>
                        <span>{p.label}</span>
                      </summary>
                      <pre className="px-3 pb-3 text-xs whitespace-pre-wrap leading-relaxed text-muted-foreground">{p.response}</pre>
                    </details>
                  ))
                ) : (
                  <p className="text-sm text-muted-foreground">Stock has no canned outputs: its criteria are upstream&apos;s.</p>
                )}
              </div>
            ))}
          </div>
        </section>

        <Legend />
      </main>
    </>
  )
}

function Cell({ arm, section }: { arm: ArmName; section?: Section }) {
  const tag = <ArmLabel arm={arm} className="lg:hidden mb-1">{ARMS.find((x) => x.id === arm)!.label}</ArmLabel>
  if (!section)
    return (
      <div>
        {tag}
        <p className="text-sm text-muted-foreground">Not part of this arm.</p>
      </div>
    )
  const multi = section.items.length > 1 || section.items.some((i) => i.tag)
  return (
    <div className="min-w-0">
      {tag}
      {section.items.length === 0 ? (
        <p className="text-sm text-muted-foreground">None.</p>
      ) : multi ? (
        <ul className="space-y-2">
          {section.items.map((item, i) => (
            <li key={i} className="text-[14px] leading-relaxed">
              {item.tag && <div className="font-mono text-[11px] text-muted-foreground">{item.tag}</div>}
              <ItemText item={item} arm={arm} />
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[14px] leading-relaxed whitespace-pre-line">
          <ItemText item={section.items[0]} arm={arm} />
        </p>
      )}
    </div>
  )
}

function Outcomes({ runs, pending }: { runs?: Execution[]; pending: boolean }) {
  if (!runs?.length) return <div className="text-xs text-muted-foreground mt-1">{pending ? 'Pending' : 'Not run'}</div>
  const notes = runs
    .filter((e) => !e.passed)
    .map((e) => {
      const what = e.passed === null ? `infrastructure (${e.infra_error ?? 'error'})` : BLAME[e.blame] ?? e.blame
      const gold = e.gold_total ? ` · ${e.gold_retrieved}/${e.gold_total} gold rows retrieved` : ''
      return `Repeat ${e.repeat + 1}: ${what}${gold}${e.stale ? ' · repeated the stock answer' : ''}`
    })
  return notes.length ? (
    <ul className="mt-1 space-y-0.5 text-xs text-muted-foreground">
      {notes.map((n) => (
        <li key={n}>{n}</li>
      ))}
    </ul>
  ) : (
    <div className="text-xs text-muted-foreground mt-1">Passed every repeat</div>
  )
}
