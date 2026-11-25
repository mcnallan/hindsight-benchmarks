'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'

// Redline shows each variant against stock; clean copy shows each arm as the model and judge read it.
export function CleanCopy({ children }: { children: React.ReactNode }) {
  const [clean, setClean] = useState(false)
  return (
    <div className={cn(clean && 'clean-copy')}>
      <div className="flex justify-end mb-3">
        <div className="inline-flex rounded-[0.625rem] border border-border p-0.5" role="group" aria-label="Text view">
          {[
            ['Redline', false],
            ['Clean copy', true],
          ].map(([label, value]) => (
            <button
              key={String(label)}
              type="button"
              aria-pressed={clean === value}
              onClick={() => setClean(value as boolean)}
              className={cn(
                'h-7 px-3 rounded-[0.5rem] font-heading font-bold text-xs transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-[hsl(var(--ring))]',
                clean === value ? 'nav-active' : 'text-muted-foreground hover:text-foreground',
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      {children}
    </div>
  )
}
