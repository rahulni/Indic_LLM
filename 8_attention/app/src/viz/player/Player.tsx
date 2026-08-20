import type { ReactNode } from 'react'
import type { Beats } from './useBeats'

export interface Beat {
  label: string
  caption: ReactNode
}

/**
 * Shared chrome for the animated scenes: transport controls, a beat scrubber, and the
 * caption panel.
 *
 * The caption is the load-bearing part. Every scene has to be comprehensible while
 * paused, with the caption carrying the point on its own - so a reader who never presses
 * play, or who has reduced motion enabled, loses nothing but the movement.
 */
export function Player({
  beats,
  ctl,
  children,
  note,
}: {
  beats: Beat[]
  ctl: Beats
  children: ReactNode
  note?: ReactNode
}) {
  const current = beats[ctl.beat]

  return (
    <div className="player">
      <div className="player-stage">{children}</div>

      <div className="player-bar">
        <button
          className="chip play"
          onClick={ctl.toggle}
          disabled={ctl.reduced}
          aria-label={ctl.playing ? 'Pause' : 'Play'}
          title={
            ctl.reduced
              ? 'Autoplay is off because your system asks for reduced motion. Use the step buttons.'
              : undefined
          }
        >
          {ctl.playing ? '❚❚' : '▶'}
        </button>
        <button className="chip" onClick={() => ctl.step(-1)} aria-label="Previous step">
          ‹
        </button>
        <button className="chip" onClick={() => ctl.step(1)} aria-label="Next step">
          ›
        </button>

        <div className="player-dots" role="tablist" aria-label="Steps">
          {beats.map((b, i) => (
            <button
              key={i}
              role="tab"
              className="player-dot"
              aria-selected={i === ctl.beat}
              aria-label={`Step ${i + 1}: ${b.label}`}
              onClick={() => ctl.goTo(i)}
            />
          ))}
        </div>

        <span className="player-step">
          {ctl.beat + 1} / {beats.length}
        </span>
      </div>

      <div className="player-caption" aria-live="polite">
        <strong>{current.label}</strong>
        <span>{current.caption}</span>
      </div>

      {ctl.reduced && (
        <p className="viz-note" style={{ marginTop: '0.5rem' }}>
          Your system asks for reduced motion, so nothing animates. Every step is still
          here &mdash; use ‹ and › to walk through it.
        </p>
      )}

      {note && (
        <p className="viz-note" style={{ marginTop: '0.6rem' }}>
          {note}
        </p>
      )}
    </div>
  )
}
