import { useMemo } from 'react'
import { attention, dot, TOKENS } from '../../lib/attention'
import { Player, type Beat } from './Player'
import { ease, lerp, useBeats } from './useBeats'

const FOCUS = 2 // "bank" - the word whose sense has to be resolved

/**
 * One word attending, animated end to end.
 *
 * Every number is computed by `attention()`; nothing here is choreographed to look
 * tidy. The beat worth watching is 4: the bars visibly shrink as raw scores become
 * percentages, but the total is pinned at exactly 100% - which is the property that
 * later forces attention sinks to exist.
 */
export function AttentionFlow() {
  const result = useMemo(() => attention(undefined, { causal: false }), [])

  const raw = useMemo(() => TOKENS.map((_, j) => dot(result.q[FOCUS], result.k[j])), [result])
  const scaled = useMemo(() => raw.map((s) => s * result.scale), [raw, result.scale])
  const weights = result.weights[FOCUS]

  const maxRaw = Math.max(...raw.map(Math.abs))

  const beats: Beat[] = [
    {
      label: 'One word asks a question',
      caption: (
        <>
          &ldquo;{TOKENS[FOCUS]}&rdquo; is ambiguous &mdash; a riverbank, or a financial
          institution? It forms a <em>query</em>: the thing it needs to know from the rest
          of the sentence.
        </>
      ),
    },
    {
      label: 'Compare it against every label',
      caption: (
        <>
          The query is multiplied against each word&rsquo;s <em>key</em>. Bigger means a
          better match. Notice every word gets a score &mdash; nothing is selected or
          fetched, which is what makes attention a soft blend rather than a lookup.
        </>
      ),
    },
    {
      label: 'Divide by √d',
      caption: (
        <>
          Every score shrinks by the same factor, 1/√{result.q[0].length} ={' '}
          {result.scale.toFixed(3)}. In high dimensions raw dot products grow large enough
          to saturate the next step, and a saturated softmax passes no gradient back.
        </>
      ),
    },
    {
      label: 'Softmax → a budget of exactly 100%',
      caption: (
        <>
          Scores become percentages: all positive, and summing to exactly 1. Watch the
          bars change size while the total does not move. That fixed total is spent{' '}
          <em>whether or not anything is worth reading</em> &mdash; which is the reason
          attention sinks had to be invented in 2023.
        </>
      ),
    },
    {
      label: 'Blend the values',
      caption: (
        <>
          Finally those percentages are applied to what each word passes on, and added
          together. The output for &ldquo;{TOKENS[FOCUS]}&rdquo; is mostly
          &ldquo;reserve&rdquo; and &ldquo;india&rdquo; &mdash; the model has resolved
          which sense of the word this is.
        </>
      ),
    },
  ]

  const ctl = useBeats(beats.length, 2000)
  const { beat, t } = ctl
  const e = ease(t)

  // Bar length as a fraction of the track, per beat. Raw and scaled share a scale so the
  // division is visible as an actual shrink; percentages then use the track as 100%.
  const frac = (j: number): number => {
    const rawF = Math.abs(raw[j]) / maxRaw
    const scaledF = Math.abs(scaled[j]) / maxRaw
    const pctF = weights[j]

    if (beat === 0) return 0
    if (beat === 1) {
      // Reveal the comparisons one at a time, left to right.
      const reached = e * TOKENS.length
      return rawF * Math.max(0, Math.min(1, reached - j))
    }
    if (beat === 2) return lerp(rawF, scaledF, e)
    if (beat === 3) return lerp(scaledF, pctF, e)
    return pctF
  }

  const shown = (j: number): string => {
    if (beat === 0) return ''
    if (beat === 1) return raw[j].toFixed(2)
    if (beat === 2) return lerp(raw[j], scaled[j], e).toFixed(2)
    if (beat === 3) {
      const v = lerp(scaled[j], weights[j] * 100, e)
      return e > 0.5 ? `${v.toFixed(0)}%` : v.toFixed(2)
    }
    return `${(weights[j] * 100).toFixed(0)}%`
  }

  const total = beat >= 3 ? weights.reduce((a, b) => a + b, 0) : null

  return (
    <Player
      beats={beats}
      ctl={ctl}
      note="Every value is computed live from the same seven-word sentence used throughout this page. Nothing is pre-rendered."
    >
      <div className="flow">
        {TOKENS.map((tok, j) => (
          <div className={`flow-row${j === FOCUS ? ' is-focus' : ''}`} key={tok}>
            <span className="flow-label">
              {tok}
              {j === FOCUS && beat === 0 && <em> ← asking</em>}
            </span>
            <span className="flow-track">
              <span
                className="flow-fill"
                style={{
                  width: `${frac(j) * 100}%`,
                  background: beat >= 3 ? 'var(--buys)' : 'var(--accent)',
                }}
              />
            </span>
            <span className="flow-num">{shown(j)}</span>
          </div>
        ))}

        {total !== null && (
          <div className="flow-total">
            <span className="flow-label">total</span>
            <span className="flow-track is-total">
              <span className="flow-fill" style={{ width: '100%', background: 'var(--buys)' }} />
            </span>
            <span className="flow-num">{(total * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>
    </Player>
  )
}
