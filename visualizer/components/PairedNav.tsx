import { SubNav } from '@/components/Page'
import { pairedCases } from '@/lib/paired'

// Overview, then every case, Retain before Reflect, so moving between cases never leaves the page header.
export function PairedNav({ current }: { current?: string }) {
  const items = [
    { href: '/paired', title: 'Overview' },
    ...pairedCases.map((c, i) => ({
      href: `/paired/${c.id}`,
      title: c.id,
      sep: i === 0 || c.operation !== pairedCases[i - 1].operation,
    })),
  ]
  return <SubNav items={items} current={current ? `/paired/${current}` : '/paired'} />
}
