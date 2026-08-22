import { useMemo, useState } from 'react'
import { attention, TOKENS } from '../lib/attention'
import { Player, type Beat } from './player/Player'
import { ease, useBeats } from './player/useBeats'

/**
 * Self-attention as relations between words, which is the picture the page was missing.
 *
 * The on-ramp used to open on three cards and a bar chart. Both are accurate and neither
 * shows the thing itself: a sentence where words reach across to each other, harder or
 * more weakly depending on how well they match. This draws that, with arc thickness taken
 * from the real softmax weights rather than chosen for looks.
 *
 * It sits before the four explanatory steps deliberately. Show the result, then explain
 * how it was computed - the steps below now answer a question the reader is already
 * holding, which is "how did it decide those arcs?".
 *
 * Causal masking is off here, as in the rest of the on-ramp, so every relation in the
 * sentence is visible. Restricting a word to looking backwards is a separate idea and is
 * introduced later, with its own toggle.
 */

const CHAR_W = 9.6
const PAD_X = 13
const GAP = 9
const ROW_Y = 34
const BOX_H = 30

export function SelfAttentionArcs() {
  const [focus, setFocus] = useState(2) // "bank" - the word whose sense is ambiguous
  const [allPairsPinned, setAllPairsPinned] = useState(false)

  const result = useMemo(() => attention(undefined, { causal: false }), [])

  // Lay the sentence out like a sentence: each word gets a box proportional to its length
  // rather than an equal slot, so it reads as text rather than as a table.
  const layout = useMemo(() => {
    let x = 4
    return TOKENS.map((t) => {
      const w = t.length * CHAR_W + PAD_X * 2
      const box = { x, w, cx: x + w / 2 }
      x += w + GAP
      return box
    })
  }, [])

  const width = layout[layout.length - 1].x + layout[layout.length - 1].w + 8
  const height = 190

  const beats: Beat[] = [
    {
      label: 'A sentence, before anything happens',
      caption: (
        <>
          Seven words. On their own they carry no relationship to each other at all &mdash;
          the model sees a bag of vectors, and nothing in it says which word belongs with
          which.
        </>
      ),
    },
    {
      label: 'One word needs to know something',
      caption: (
        <>
          &ldquo;{TOKENS[focus]}&rdquo; is ambiguous on its own. Is it a riverbank, or a
          financial institution? It cannot tell from itself; the answer is somewhere else in
          the sentence.
        </>
      ),
    },
    {
      label: 'It reaches out to every other word',
      caption: (
        <>
          One arc per word, and the thickness is how much attention it gets. These are the
          real numbers &mdash; &ldquo;{TOKENS[focus]}&rdquo; leans hardest on{' '}
          <strong>{strongest(result.weights[focus], focus)}</strong>. That is the model
          resolving which sense of the word this is.
        </>
      ),
    },
    {
      label: 'Every word does this at the same time',
      caption: (
        <>
          Each word&rsquo;s strongest link, all at once. Nothing here happened in sequence:
          every one of these was computed in the same step, which is exactly what a
          recurrent network could not do and why this idea took over.
        </>
      ),
    },
  ]

  const ctl = useBeats(beats.length, 2300)
  const { beat, t } = ctl

  const allPairs = allPairsPinned || beat === 3
  const showArcs = beat >= 2 || allPairsPinned
  const reveal = beat === 2 ? ease(t) * TOKENS.length : TOKENS.length

  /** Arcs to draw: either the focus word's fan, or each word's single best link. */
  const arcs = useMemo(() => {
    if (allPairs) {
      return TOKENS.map((_, i) => {
        const row = result.weights[i]
        let best = -1
        let bestW = -1
        row.forEach((w, j) => {
          if (j !== i && w > bestW) {
            bestW = w
            best = j
          }
        })
        return { from: i, to: best, w: bestW }
      })
    }
    return TOKENS.map((_, j) => ({ from: focus, to: j, w: result.weights[focus][j] })).filter(
      (a) => a.to !== focus,
    )
  }, [allPairs, focus, result])

  return (
    <div>
      <div className="controls">
        <span style={{ fontSize: '0.8rem', color: 'var(--ink-faint)' }}>Follow a word:</span>
        {TOKENS.map((t2, i) => (
          <button
            key={t2}
            className="chip"
            aria-pressed={focus === i && !allPairs}
            onClick={() => {
              setFocus(i)
              setAllPairsPinned(false)
            }}
          >
            {t2}
          </button>
        ))}
        <button
          className="chip"
          aria-pressed={allPairsPinned}
          onClick={() => setAllPairsPinned((v) => !v)}
        >
          every word at once
        </button>
      </div>

      <Player
        beats={beats}
        ctl={ctl}
        note="Arc thickness is the real attention weight, computed live — not drawn to look tidy."
      >
        <div className="scroll-x">
          <svg
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label="The sentence with arcs showing how strongly each word attends to the others"
          >
            {showArcs &&
              arcs.map((a, idx) => {
                if (a.to < 0) return null
                if (!allPairs && idx >= reveal) return null

                const x1 = layout[a.from].cx
                const x2 = layout[a.to].cx
                const span = Math.abs(x2 - x1)
                const dip = Math.min(30 + span * 0.42, height - ROW_Y - BOX_H - 16)
                const y0 = ROW_Y + BOX_H
                const cy = y0 + dip

                return (
                  <path
                    key={`${a.from}-${a.to}`}
                    d={`M ${x1} ${y0} Q ${(x1 + x2) / 2} ${cy} ${x2} ${y0}`}
                    fill="none"
                    stroke={allPairs ? 'var(--cat-memory)' : 'var(--accent)'}
                    strokeWidth={1 + a.w * 11}
                    strokeLinecap="round"
                    opacity={0.22 + a.w * 0.78}
                  />
                )
              })}

            {TOKENS.map((t2, i) => {
              const isFocus = !allPairs && i === focus && beat >= 1
              return (
                <g
                  key={t2}
                  onClick={() => {
                    setFocus(i)
                    setAllPairsPinned(false)
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  <rect
                    x={layout[i].x}
                    y={ROW_Y}
                    width={layout[i].w}
                    height={BOX_H}
                    rx="6"
                    fill={isFocus ? 'var(--accent)' : 'var(--bg-sunken)'}
                    stroke={isFocus ? 'var(--accent)' : 'var(--border-strong)'}
                  />
                  <text
                    x={layout[i].cx}
                    y={ROW_Y + BOX_H / 2 + 5}
                    textAnchor="middle"
                    fontSize="14"
                    fontFamily="var(--mono)"
                    fill={isFocus ? 'var(--bg)' : 'var(--ink)'}
                  >
                    {t2}
                  </text>
                </g>
              )
            })}

            {beat === 1 && !allPairs && (
              <text
                x={layout[focus].cx}
                y={ROW_Y - 12}
                textAnchor="middle"
                fontSize="11"
                fill="var(--accent)"
                fontFamily="var(--mono)"
              >
                ? which sense am I ?
              </text>
            )}
          </svg>
        </div>
      </Player>
    </div>
  )
}

/** The word a query leans on hardest, ignoring itself. */
function strongest(row: number[], self: number): string {
  let best = 0
  let bestW = -1
  row.forEach((w, j) => {
    if (j !== self && w > bestW) {
      bestW = w
      best = j
    }
  })
  return TOKENS[best]
}
