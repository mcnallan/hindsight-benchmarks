import type { Row, TokenTotals } from '@/lib/data'
import { cn } from '@/lib/utils'

// Shared by server pages and the client board, so no 'use client' here.
export const fmtCost = (v: number | null) => (v === null ? '—' : v === 0 ? 'free' : v < 0.01 ? '<$0.01' : `$${v < 10 ? v.toFixed(2) : v.toFixed(0)}`)
export const fmtJudge = (judge: string) => judge.split('/').pop()?.replace(':', ' ')
export const fmtTokens = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : String(v))

// The hover text for a run's token totals: every count in full, reasoning split out of output.
export function tokenTip(t: TokenTotals) {
  const n = (v: number) => v.toLocaleString('en-US')
  const lines = [
    `Totals over ${n(t.cases)} cases`,
    `LLM calls: ${n(t.calls)}`,
    `Input tokens: ${n(t.input)}${t.cached !== undefined ? ` (${n(t.cached)} cached)` : ''}`,
    `Output tokens: ${n(t.output)}`,
    t.reasoning !== undefined
      ? `  answer: ${n(t.output - t.reasoning)}\n  reasoning: ${n(t.reasoning)}`
      : '  reasoning: not reported separately',
  ]
  if (t.decisionCalls) lines.push(`Decision calls: ${n(t.decisionCalls)}`, `Decision input tokens: ${n(t.decisionInput ?? 0)}`)
  return lines.join('\n')
}

// Rows that carry no icon of their own get their provider's logo.
export const PROVIDER_ICON: Record<string, string> = {
  BAAI: '/logos/BAAI.webp',
  NVIDIA: '/logos/nvidia.webp',
  SBERT: '/logos/sentence-transformers.webp',
  'cross-encoder': '/logos/cross-encoder.webp',
  'JHU CLSP': '/logos/jhu-clsp.webp',
  Microsoft: '/logos/microsoft.webp',
  IBM: '/logos/ibm.webp',
  Snowflake: '/logos/snowflake.webp',
  Alibaba: '/logos/alibaba.webp',
  Mixedbread: '/logos/mixedbread.webp',
  FlashRank: '/logos/cross-encoder.webp',
  Hindsight: '/favicon.png',
  'Hindsight without a reranker': '/favicon.png',
  Cohere: '/cohere.png',
  'Jina AI': '/jina.png',
  'Voyage AI': '/voyageai.png',
  OpenAI: '/openai.png',
  Google: '/gemini.png',
  ZeroEntropy: '/zeroentropy.png',
}

export function ModelCell({ row }: { row: Pick<Row, 'model' | 'provider' | 'icon' | 'openWeights' | 'tag' | 'reasoningMode' | 'split'> & { size?: string } }) {
  return (
    <div className="flex items-center gap-2.5 min-w-0">
      {(row.icon ?? PROVIDER_ICON[row.provider]) ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={row.icon ?? PROVIDER_ICON[row.provider]} alt="" className="w-5 h-5 object-contain shrink-0 rounded-sm" />
      ) : (
        <span className="w-5 h-5 shrink-0 rounded-sm bg-secondary text-[10px] font-bold flex items-center justify-center text-muted-foreground">
          {row.provider[0]}
        </span>
      )}
      <div className="min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[13px] font-medium truncate">{row.model}</span>
          {row.tag === 'reference' && <Tag className="text-[hsl(var(--teal))] border-[hsl(var(--teal))]/40">reference</Tag>}
          {row.tag === 'baseline' && <Tag className="text-muted-foreground border-border">baseline</Tag>}
          {row.tag === 'published' && <Tag className="text-muted-foreground border-border">published</Tag>}
        </div>
        {row.split && (
          <div className="font-mono text-[12px] font-medium truncate">
            +{' '}
            <span className="underline decoration-2 underline-offset-[3px] text-[hsl(var(--primary))]">{row.split.decisionModel}</span>
          </div>
        )}
        <div className="text-xs text-muted-foreground truncate">
          {row.provider}
          {row.reasoningMode && ` · ${row.reasoningMode}`}
          {row.size && ` · ${row.size}`}
          {row.openWeights && ' · open weights'}
        </div>
      </div>
    </div>
  )
}

function Tag({ children, className }: { children: React.ReactNode; className?: string }) {
  return <span className={cn('text-[10px] font-semibold uppercase tracking-wide border rounded px-1 py-px', className)}>{children}</span>
}

