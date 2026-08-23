import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

/**
 * How much the reader already knows, and what the page does about it.
 *
 * This replaced a binary plain/technical toggle that swapped exactly two strings. That
 * was not a proficiency level: someone new still met all fifty-two entries with every
 * diagram collapsed and full citation furniture, which is most of what actually
 * overwhelms a beginner. The level now decides *composition* - how much is shown, which
 * register it is written in, and what is expanded by default.
 *
 * Two rules this must never break:
 *
 *  - Nothing is locked. The level sets defaults; the path chips and search still reach
 *    every mechanism from any level.
 *  - Substance never hides. Trade-offs and dates render at all three levels. Only
 *    emphasis, ordering and default expansion change, because a page that hid what a
 *    technique costs would be worse for a beginner, not kinder to them.
 */
export type Level = 'new' | 'building' | 'research'

export interface LevelConfig {
  /** Which prose field the cards render. */
  prose: 'plain' | 'mechanism'
  /** Show the core equation, where one exists. */
  equation: boolean
  /** Show the plain-language analogy boxes in the on-ramp. */
  analogy: boolean
  /** Open each card's diagram without being asked. */
  diagramOpen: boolean
  /** Show the check-yourself questions. */
  checks: boolean
  /** Underline glossary terms. Definitions stay reachable either way. */
  glossary: boolean
  /** Show arXiv identifiers and the date-evidence line. */
  sources: boolean
  /** Show the "easy to get wrong" naming traps. */
  traps: boolean
  /** Which mechanisms are listed by default. */
  scope: 'essentials' | 'all'
}

export const LEVELS: { id: Level; label: string; blurb: string }[] = [
  { id: 'new', label: 'New to this', blurb: 'Start from what a word is even doing' },
  { id: 'building', label: 'I build with these', blurb: 'All 52, with the trade-offs' },
  { id: 'research', label: 'I read the papers', blurb: 'Technical prose and equations' },
]

export const LEVEL_CONFIG: Record<Level, LevelConfig> = {
  new: {
    prose: 'plain',
    equation: false,
    analogy: true,
    diagramOpen: true,
    checks: true,
    glossary: true,
    sources: false,
    traps: false,
    scope: 'essentials',
  },
  building: {
    prose: 'plain',
    equation: false,
    analogy: false,
    diagramOpen: false,
    checks: true,
    glossary: true,
    sources: true,
    traps: true,
    scope: 'all',
  },
  research: {
    prose: 'mechanism',
    equation: true,
    analogy: false,
    diagramOpen: false,
    checks: false,
    glossary: false,
    sources: true,
    traps: true,
    scope: 'all',
  },
}

const KEY = 'attention-level'

const LevelCtx = createContext<{
  level: Level
  config: LevelConfig
  setLevel: (l: Level) => void
}>({ level: 'building', config: LEVEL_CONFIG.building, setLevel: () => {} })

/**
 * `initial` exists so the smoke test can render every level. In the app it is left
 * unset and the stored preference (or 'building') wins.
 */
export function LevelProvider({
  children,
  initial = 'building',
}: {
  children: ReactNode
  initial?: Level
}) {
  const [level, setLevel] = useState<Level>(initial)

  // Read the stored choice after mount rather than during render, so the server render
  // and the first client render agree.
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(KEY)
      if (stored === 'new' || stored === 'building' || stored === 'research') {
        setLevel(stored)
      }
    } catch {
      // Blocked storage. The default is fine.
    }
  }, [])

  useEffect(() => {
    try {
      window.localStorage.setItem(KEY, level)
    } catch {
      // Not remembering the choice is not worth breaking the page over.
    }
  }, [level])

  return (
    <LevelCtx.Provider value={{ level, config: LEVEL_CONFIG[level], setLevel }}>
      {children}
    </LevelCtx.Provider>
  )
}

export const useLevel = () => useContext(LevelCtx)

/** The three chips, used in both the header and the hero. */
export function LevelPicker({ compact = false }: { compact?: boolean }) {
  const { level, setLevel } = useLevel()

  if (compact) {
    return (
      <>
        {LEVELS.map((l) => (
          <button
            key={l.id}
            className="navbtn level-btn"
            aria-current={level === l.id}
            onClick={() => setLevel(l.id)}
            title={l.blurb}
          >
            {l.label}
          </button>
        ))}
      </>
    )
  }

  return (
    <div className="level-picker">
      <span className="level-ask">How much do you already know?</span>
      <div className="level-options">
        {LEVELS.map((l) => (
          <button
            key={l.id}
            className="level-chip"
            aria-pressed={level === l.id}
            onClick={() => setLevel(l.id)}
            title={l.blurb}
          >
            {l.label}
          </button>
        ))}
      </div>
      <span className="level-note">Changes how much is shown. Nothing is locked away.</span>
    </div>
  )
}
