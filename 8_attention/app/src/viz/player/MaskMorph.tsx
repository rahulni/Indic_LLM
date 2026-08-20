import { useMemo } from 'react'
import { attention, buildMask, maskDensity, type MaskKind } from '../../lib/attention'
import { Player, type Beat } from './Player'
import { ease, lerp, useBeats } from './useBeats'

const N = 22
const CELL = 13

interface Pattern {
  kind: MaskKind
  label: string
  who: string
  rule: string
}

/**
 * The sparse family, morphing from one pattern into the next.
 *
 * Presented as a single continuous shape on purpose. Six years of sparse-attention
 * research reads like a dozen unrelated inventions until you watch them deform into one
 * another, at which point it is obvious they are one move - restrict which keys a query
 * may read - with different rules for choosing. Cells fade in and out between beats, so
 * what changed is visible rather than something you have to diff by eye.
 *
 * Reuses `buildMask` and `maskDensity` unchanged, so the density readout is the real
 * fraction of the causal matrix each pattern computes.
 */
const PATTERNS: Pattern[] = [
  {
    kind: 'full',
    label: 'Full attention',
    who: 'Transformer, 2017',
    rule: 'Every word reads every earlier word. Exact, and quadratic — the bill all of the following are paying down.',
  },
  {
    kind: 'strided',
    label: 'Strided',
    who: 'Sparse Transformers, 2019',
    rule: 'Neighbours, plus every few words so news can still cross the sequence in two hops. Drawn by a human in advance, so it cannot know which word actually mattered.',
  },
  {
    kind: 'window',
    label: 'Sliding window',
    who: 'Longformer 2020 · Mistral 2023',
    rule: 'Only the last few words. Memory stops growing; anything older is invisible in this layer and has to arrive via depth instead.',
  },
  {
    kind: 'window-sink',
    label: 'Window + sinks',
    who: 'StreamingLLM, 2023',
    rule: 'The window, plus the first few words pinned permanently. Four extra cells, and the model can stream indefinitely instead of collapsing.',
  },
  {
    kind: 'topk',
    label: 'Top-k',
    who: 'Top-k 2021 · DSA 2025',
    rule: 'The first rule here chosen by content rather than position — keep the best-matching keys. The catch is that you must score everything before you can rank it.',
  },
  {
    kind: 'block-compressed',
    label: 'Compressed + selected blocks',
    who: 'NSA, 2025',
    rule: 'Coarse block summaries for reach, a fine local window for detail, and the model trained to work this way from the start rather than having it imposed afterwards.',
  },
]

export function MaskMorph() {
  // Real scores, so the content-dependent pattern selects on merit rather than on an
  // invented ranking. The toy sentence is short, so it tiles to fill the grid.
  const scores = useMemo(() => {
    const base = attention().rawScores
    const m = base.length
    return Array.from({ length: N }, (_, i) =>
      Array.from({ length: N }, (_, j) => base[i % m][j % m] + (i === j ? 0.4 : 0)),
    )
  }, [])

  const masks = useMemo(
    () =>
      PATTERNS.map((p) =>
        buildMask(N, p.kind, {
          windowSize: 5,
          stride: 4,
          sinks: 2,
          topK: 5,
          blockSize: 4,
          scores,
        }),
      ),
    [scores],
  )

  const densities = useMemo(() => masks.map(maskDensity), [masks])

  const beats: Beat[] = PATTERNS.map((p, i) => ({
    label: p.label,
    caption: (
      <>
        <em>{p.who}.</em> {p.rule} Computes {(densities[i] * 100).toFixed(0)}% of the full
        causal matrix.
      </>
    ),
  }))

  const ctl = useBeats(beats.length, 2400)
  const { beat, t } = ctl

  // Cross-fade into the next pattern during the second half of each beat, so a cell that
  // appears or disappears is seen doing it.
  const nextBeat = (beat + 1) % PATTERNS.length
  const blend = ctl.playing ? ease(Math.max(0, (t - 0.55) / 0.45)) : 0

  const from = masks[beat]
  const to = masks[nextBeat]
  const density = lerp(densities[beat], densities[nextBeat], blend)

  const size = N * CELL

  return (
    <Player
      beats={beats}
      ctl={ctl}
      note="Filled cells are computed; empty ones are skipped entirely. Everything above the diagonal is masked by causality in all six — a word can never read its own future."
    >
      <div className="morph">
        <div className="scroll-x">
          <svg
            width={size + 2}
            height={size + 2}
            viewBox={`0 0 ${size + 2} ${size + 2}`}
            role="img"
            aria-label={`${PATTERNS[beat].label} attention pattern`}
          >
            {from.map((row, i) =>
              row.map((wasOn, j) => {
                if (j > i) return null
                const willBeOn = to[i][j]
                const opacity = lerp(wasOn ? 0.85 : 0.06, willBeOn ? 0.85 : 0.06, blend)
                return (
                  <rect
                    key={`${i}-${j}`}
                    x={j * CELL + 1}
                    y={i * CELL + 1}
                    width={CELL - 1.5}
                    height={CELL - 1.5}
                    rx="1.5"
                    fill="var(--accent)"
                    opacity={opacity}
                  />
                )
              }),
            )}
          </svg>
        </div>

        <div className="readout" style={{ marginTop: '0.8rem' }}>
          <div>
            <span>cells computed</span>
            <span className="em">{(density * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span>skipped</span>
            <span className="em">{((1 - density) * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span>rule</span>
            <span>{beat === 4 || beat === 5 ? 'chosen by content' : 'drawn in advance'}</span>
          </div>
        </div>
      </div>
    </Player>
  )
}
