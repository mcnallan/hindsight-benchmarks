import { cn } from '@/lib/utils'
import type { ArmName, Execution, Item, Segment } from '@/lib/paired'

// Arm identity: stock is the neutral page, wording takes the brand blue, a counterfactual takes
// the amber highlighter that also marks its changed fact.
export const ARM_TONE: Record<ArmName, string> = {
  stock: 'text-muted-foreground',
  wording: 'text-[hsl(var(--primary))]',
  counterfactual: 'text-[hsl(var(--cf))]',
}

export function ArmLabel({ arm, children, className }: { arm: ArmName; children: React.ReactNode; className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-1.5 font-heading font-bold text-sm', ARM_TONE[arm], className)}>
      <i className={cn('inline-block w-2 h-2', `arm-dot-${arm}`)} aria-hidden />
      {children}
    </span>
  )
}

// The redline: what stock said struck through, what this arm says marked in its own colour.
export function Redline({ segments, arm }: { segments: Segment[]; arm: ArmName }) {
  return (
    <>
      {segments.map((s, i) =>
        s.op === 'eq' ? (
          <span key={i}>{s.text}</span>
        ) : s.op === 'del' ? (
          <del key={i} className="rl-del">
            {s.text}
          </del>
        ) : (
          <ins key={i} className={arm === 'counterfactual' ? 'rl-ins-cf' : 'rl-ins'}>
            {s.text}
          </ins>
        ),
      )}
    </>
  )
}

export function ItemText({ item, arm }: { item: Item; arm: ArmName }) {
  return item.diff ? <Redline segments={item.diff} arm={arm} /> : <>{item.text}</>
}

// One square per repeat: filled for a pass, outlined red for a fail, hatched for an
// infrastructure failure, dotted while it has not run.
export function RepeatMarks({ runs, repeats, label }: { runs?: Execution[]; repeats: number; label: string }) {
  return (
    <span className="inline-flex gap-[3px]" role="img" aria-label={`${label}: ${describe(runs, repeats)}`}>
      {Array.from({ length: repeats }, (_, r) => {
        const e = runs?.find((x) => x.repeat === r)
        const state = !e ? 'pending' : e.passed === null ? 'infra' : e.passed ? 'pass' : 'fail'
        return (
          <i
            key={r}
            title={e ? `repeat ${r + 1}: ${state === 'infra' ? `infrastructure (${e.infra_error ?? 'error'})` : state}${e.blame && !e.passed ? ` · ${e.blame}` : ''}${e.stale ? ' · repeated the stock answer' : ''}` : `repeat ${r + 1}: not run`}
            className={cn('mark', `mark-${state}`, e?.stale && 'mark-stale')}
          />
        )
      })}
    </span>
  )
}

function describe(runs: Execution[] | undefined, repeats: number) {
  if (!runs?.length) return 'not run'
  const graded = runs.filter((e) => e.passed !== null)
  const infra = runs.length - graded.length
  return `${graded.filter((e) => e.passed).length} of ${graded.length} passed${infra ? `, ${infra} infrastructure` : ''}${runs.length < repeats ? `, ${repeats - runs.length} not run` : ''}`
}

export function Legend() {
  return (
    <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-1.5"><i className="mark mark-pass" /> Passed</span>
      <span className="inline-flex items-center gap-1.5"><i className="mark mark-fail" /> Failed</span>
      <span className="inline-flex items-center gap-1.5"><i className="mark mark-fail mark-stale" /> Repeated the stock answer</span>
      <span className="inline-flex items-center gap-1.5"><i className="mark mark-infra" /> Infrastructure failure</span>
      <span className="inline-flex items-center gap-1.5"><i className="mark mark-pending" /> Not run yet</span>
      <span className="inline-flex items-center gap-1.5"><del className="rl-del">stock</del> <ins className="rl-ins">wording</ins> <ins className="rl-ins-cf">counterfactual</ins></span>
    </div>
  )
}
