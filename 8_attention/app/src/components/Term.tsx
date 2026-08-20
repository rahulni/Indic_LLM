import { useId, useState } from 'react'
import glossary from '../data/glossary.json'
import { useLevel } from './LevelContext'

const GLOSSARY = glossary as Record<string, string>

/**
 * An inline glossary term.
 *
 * Built as a real <button> rather than a CSS :hover tooltip so it can be reached by
 * keyboard and announced by a screen reader - a definition nobody can open is not a
 * definition. Pointer users get it on hover, keyboard users on focus, touch users on tap.
 *
 * An unknown key renders as plain text instead of throwing, and the smoke test asserts
 * that every key used on the page actually resolves, so a typo is caught by a command
 * rather than by silently swallowing the definition.
 */
export function Term({ k, children }: { k: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const { config } = useLevel()
  const definition = GLOSSARY[k]

  // Someone who reads the papers does not need "softmax" underlined on every page.
  // The definition stays reachable at every level; only the visual marker goes away.
  if (!definition || !config.glossary) return <>{children}</>

  return (
    <span className="term-wrap">
      <button
        type="button"
        className="term"
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
      >
        {children}
      </button>
      {open && (
        <span className="term-pop" id={id} role="tooltip">
          {definition}
        </span>
      )}
    </span>
  )
}

export { GLOSSARY }
