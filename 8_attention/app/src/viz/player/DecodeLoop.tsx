import { useState } from 'react'
import { Player, type Beat } from './Player'
import { Formula } from '../../components/Formula'
import { useBeats } from './useBeats'

const WORDS = [
  'The', 'reserve', 'bank', 'of', 'india', 'raised', 'rates',
  'by', 'a', 'quarter', 'of', 'a', 'point', 'today',
]

const WINDOW = 5
const SINKS = 2

type Mode = 'full' | 'window' | 'sink' | 'linear'

const MODES: { id: Mode; label: string; who: string }[] = [
  { id: 'full', label: 'Full attention', who: 'Transformer, 2017' },
  { id: 'window', label: 'Sliding window', who: 'Longformer 2020 · Mistral 2023' },
  { id: 'sink', label: 'Window + sinks', who: 'StreamingLLM, 2023' },
  { id: 'linear', label: 'Linear state', who: 'DeltaNet · Mamba · GSA' },
]

/**
 * Writing text one word at a time, with the stored keys and values shown underneath.
 *
 * This is the hardest idea on the page to convey with a still image, because the whole
 * point is what happens *over time*: whether the stored memory keeps growing, plateaus,
 * or never grows at all. Switching modes with the same word sequence makes four different
 * mechanisms directly comparable.
 *
 * Deliberately no quality or perplexity numbers anywhere here. The animation shows
 * structure - what is kept and what is dropped - and where the consequence is qualitative
 * the caption says so in words rather than inventing a metric to dramatise it.
 */
export function DecodeLoop() {
  const [mode, setMode] = useState<Mode>('full')
  const ctl = useBeats(WORDS.length, 1100)
  const step = ctl.beat

  /** Which token positions still have their keys and values stored at this step. */
  const kept = (): number[] => {
    const all = Array.from({ length: step + 1 }, (_, i) => i)
    switch (mode) {
      case 'full':
        return all
      case 'window':
        return all.filter((i) => step - i < WINDOW)
      case 'sink':
        return all.filter((i) => step - i < WINDOW || i < SINKS)
      case 'linear':
        return [] // one fixed state, drawn separately - there are no per-token blocks
    }
  }

  const stored = kept()
  const blockCount = mode === 'linear' ? 1 : stored.length

  const caption = (): Beat['caption'] => {
    const early = step < WINDOW
    switch (mode) {
      case 'full':
        return early ? (
          <>
            Each word that gets written leaves behind a key and a value, and they are kept
            so they need not be recomputed. Nothing is ever discarded.
          </>
        ) : (
          <>
            Still growing, one block per word, and it never stops. At 128,000 words this is
            gigabytes &mdash; and that is for <em>one</em> conversation. Every simultaneous
            user needs their own.
          </>
        )
      case 'window':
        return early ? (
          <>
            Filling up the window. So far this looks identical to full attention.
          </>
        ) : (
          <>
            The window is full, so each new word pushes the oldest one out. Memory has
            stopped growing &mdash; that is the win. But &ldquo;The&rdquo; and
            &ldquo;reserve&rdquo; are gone from this layer now, and no later word can look
            at them directly.
          </>
        )
      case 'sink':
        return early ? (
          <>
            Filling up, with the first {SINKS} blocks marked to keep permanently. Watch what
            happens once the window fills.
          </>
        ) : (
          <>
            The window rolls as before, but the first {SINKS} blocks stay pinned. Remember
            that attention must spend exactly 100% somewhere: when a word has nothing it
            wants, it dumps that budget on the earliest tokens. Evict those and every
            distribution is forced onto words nobody chose &mdash; which is why removing
            them breaks a model that otherwise streams happily forever.
          </>
        )
      case 'linear':
        return (
          <>
            No per-word blocks at all. One fixed-size state is updated in place with every
            word, so memory is identical at word 1 and word 1,000,000. What is given up is
            exactness: the state is a blend, so no specific earlier word can be recovered
            from it intact.
          </>
        )
    }
  }

  // Which term of the cache expression this mode is acting on, and what it does to it.
  const FORMULA_FOR: Record<Mode, { part: string; gloss: string }> = {
    full: {
      part: 'tokens',
      gloss:
        'T is every token ever written, and it never stops climbing. This is the term the three mechanisms below each attack.',
    },
    window: {
      part: 'tokens',
      gloss:
        'T is capped at the window size w. The cache stops growing entirely - and everything older than w is gone from this layer.',
    },
    sink: {
      part: 'tokens',
      gloss:
        'T becomes w + s: the rolling window plus a handful of permanently pinned tokens. Four extra entries buys unlimited streaming.',
    },
    linear: {
      part: 'tokens',
      gloss:
        'There is no T at all. A fixed-size state replaces the whole expression, so memory is the same at one token and at a million - and no single earlier token can be recovered exactly.',
    },
  }

  const beats: Beat[] = WORDS.map((w, i) => ({
    label: `Writing word ${i + 1}: “${w}”`,
    caption: caption(),
  }))

  return (
    <div>
      <div className="controls">
        {MODES.map((m) => (
          <button
            key={m.id}
            className="chip"
            aria-pressed={mode === m.id}
            onClick={() => setMode(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      <Player
        beats={beats}
        ctl={ctl}
        note={
          <>
            Same sentence in every mode, so the difference you see is the mechanism and
            nothing else. {MODES.find((m) => m.id === mode)!.who}.
          </>
        }
      >
        <div className="decode">
          <div className="decode-words">
            {WORDS.map((w, i) => (
              <span
                key={i}
                className={`decode-word${i <= step ? ' is-written' : ''}${
                  i === step ? ' is-current' : ''
                }`}
              >
                {w}
              </span>
            ))}
          </div>

          <div className="decode-label">
            stored keys &amp; values
            <span className="decode-count">
              {mode === 'linear' ? '1 state, forever' : `${blockCount} block${blockCount === 1 ? '' : 's'}`}
            </span>
          </div>

          <div className="decode-blocks">
            {mode === 'linear' ? (
              <span className="decode-block is-state">
                state
                <em>updated in place</em>
              </span>
            ) : (
              WORDS.map((w, i) => {
                if (i > step) return <span key={i} className="decode-block is-future" />
                const isKept = stored.includes(i)
                const isSink = mode === 'sink' && i < SINKS
                return (
                  <span
                    key={i}
                    className={`decode-block${isKept ? ' is-kept' : ' is-dropped'}${
                      isSink ? ' is-sink' : ''
                    }`}
                    title={`${w}${isKept ? '' : ' — evicted'}`}
                  >
                    {isKept ? w.slice(0, 4) : '·'}
                  </span>
                )
              })
            )}
          </div>

          {mode === 'sink' && (
            <p className="decode-key">
              <span className="swatch is-sink" /> pinned sinks &nbsp;
              <span className="swatch is-kept" /> rolling window &nbsp;
              <span className="swatch is-dropped" /> evicted
            </p>
          )}
        </div>

        <Formula
          id="kvCache"
          active={FORMULA_FOR[mode].part}
          glossOverride={FORMULA_FOR[mode].gloss}
        />
      </Player>
    </div>
  )
}
