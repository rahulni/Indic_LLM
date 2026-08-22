import { FORMULAS, glossFor } from '../data/formulas'

/**
 * A formula with one term lit up, and that term explained underneath.
 *
 * The highlight is the whole point. A formula printed next to an animation is decoration;
 * a formula whose active term moves in step with what the animation is doing is an
 * explanation. Stepping through AttentionFlow walks this highlight from Q K^T to the
 * scaling to the softmax to V, in time with the bars doing each of those things.
 *
 * Rendered at every proficiency level. Formulas only intimidate when they arrive
 * unexplained, and here every symbol has a picture beside it and a gloss below it.
 */
export function Formula({
  id,
  active = null,
  glossOverride,
  compact = false,
}: {
  id: keyof typeof FORMULAS
  /** Which part key to highlight. Parts sharing a key highlight together. */
  active?: string | null
  /** Replaces the part's own gloss, for scenes where context changes the meaning. */
  glossOverride?: string
  compact?: boolean
}) {
  const def = FORMULAS[id]
  if (!def) return null

  const gloss = glossOverride ?? glossFor(def, active)

  return (
    <div className={`formula${compact ? ' is-compact' : ''}`}>
      <div className="formula-expr scroll-x">
        <code>
          {def.parts.map((p, i) => (
            <span
              key={i}
              className={`fpart${active && p.key === active ? ' is-active' : ''}`}
            >
              {p.text}
            </span>
          ))}
        </code>
      </div>

      {gloss ? (
        <p className="formula-gloss" aria-live="polite">
          <span className="formula-arrow" aria-hidden="true">
            ▸
          </span>
          {gloss}
        </p>
      ) : def.note ? (
        <p className="formula-note">{def.note}</p>
      ) : null}
    </div>
  )
}

/** Two formulas side by side, for comparing an update rule against its fix. */
export function FormulaPair({
  left,
  right,
  leftLabel,
  rightLabel,
  activeSide,
}: {
  left: keyof typeof FORMULAS
  right: keyof typeof FORMULAS
  leftLabel: string
  rightLabel: string
  activeSide: 'left' | 'right' | null
}) {
  return (
    <div className="formula-pair">
      <div className={`formula-side${activeSide === 'left' ? ' is-on' : ''}`}>
        <span className="formula-label">{leftLabel}</span>
        <Formula id={left} active={activeSide === 'left' ? 'add' : null} compact />
      </div>
      <div className={`formula-side${activeSide === 'right' ? ' is-on' : ''}`}>
        <span className="formula-label">{rightLabel}</span>
        <Formula id={right} active={activeSide === 'right' ? 'erase' : null} compact />
      </div>
    </div>
  )
}
