import { useMemo, useState } from 'react'
import { attention, buildMask, maskDensity, type MaskKind } from '../lib/attention'

const N = 24
const CELL = 13

interface PatternDef {
  kind: MaskKind
  label: string
  who: string
  note: string
}

/**
 * Every sparse mechanism on one grid.
 *
 * Presenting them together is the argument: the sparse family is not a dozen unrelated
 * inventions, it is one move - restrict which keys a query may read - with different
 * rules for choosing. The rules divide sharply into those drawn in advance by a human
 * and the two that look at content, which is the axis the field spent six years
 * crossing.
 */
const PATTERNS: PatternDef[] = [
  {
    kind: 'causal',
    label: 'Full causal',
    who: 'Transformer, 2017',
    note: 'Every token sees every earlier token. Exact, and quadratic — the bill everything below is trying to reduce.',
  },
  {
    kind: 'strided',
    label: 'Strided',
    who: 'Sparse Transformers, 2019',
    note: 'Local neighbours plus every k-th token, so information routes globally in two hops. Content-blind: it cannot know which token mattered.',
  },
  {
    kind: 'window',
    label: 'Sliding window',
    who: 'Longformer 2020 · Mistral 2023',
    note: 'Only the last w tokens. The KV cache stops growing, but anything older is invisible in this layer — depth is what recovers it, lossily.',
  },
  {
    kind: 'window-sink',
    label: 'Window + sinks',
    who: 'StreamingLLM, 2023',
    note: 'The window, plus the first few tokens pinned forever. Softmax must put its mass somewhere; evict those and every distribution is forced onto tokens it never wanted.',
  },
  {
    kind: 'topk',
    label: 'Top-k (content-aware)',
    who: 'Top-k 2021 · DSA 2025',
    note: 'Keep the k highest-scoring keys per query. The first pattern here chosen by content rather than by position — but you must score everything before you can rank it.',
  },
  {
    kind: 'block-compressed',
    label: 'Compressed + selected blocks',
    who: 'NSA, 2025',
    note: 'Coarse block summaries for global reach, plus a fine local window. Trained this way from the start, so the model learns to work with sparsity instead of having it imposed afterwards.',
  },
]

export function MaskPatterns() {
  const [active, setActive] = useState<MaskKind>('causal')
  const [windowSize, setWindowSize] = useState(5)

  // Real scores drive the content-dependent patterns, so top-k selects on merit rather
  // than on an invented ranking. The toy sentence is short, so it is tiled to fill N.
  const scores = useMemo(() => {
    const base = attention().rawScores
    const m = base.length
    return Array.from({ length: N }, (_, i) =>
      Array.from({ length: N }, (_, j) => base[i % m][j % m] + (i === j ? 0.4 : 0)),
    )
  }, [])

  const def = PATTERNS.find((p) => p.kind === active)!

  const mask = useMemo(
    () =>
      buildMask(N, active, {
        windowSize,
        stride: 4,
        sinks: 2,
        topK: 5,
        blockSize: 4,
        scores,
      }),
    [active, windowSize, scores],
  )

  const density = maskDensity(mask)
  const size = N * CELL

  return (
    <div className="viz-wrap">
      <div>
        <div className="controls">
          {PATTERNS.map((p) => (
            <button
              key={p.kind}
              className="chip"
              aria-pressed={active === p.kind}
              onClick={() => setActive(p.kind)}
            >
              {p.label}
            </button>
          ))}
        </div>

        {(active === 'window' || active === 'window-sink') && (
          <div className="controls">
            <label className="field">
              window size: {windowSize}
              <input
                type="range"
                min={2}
                max={12}
                value={windowSize}
                onChange={(e) => setWindowSize(Number(e.target.value))}
              />
            </label>
          </div>
        )}

        <div className="scroll-x">
          <svg
            width={size + 2}
            height={size + 2}
            viewBox={`0 0 ${size + 2} ${size + 2}`}
            role="img"
            aria-label={`${def.label} attention pattern over ${N} tokens`}
          >
            {mask.map((row, i) =>
              row.map((allowed, j) => (
                <rect
                  key={`${i}-${j}`}
                  x={j * CELL + 1}
                  y={i * CELL + 1}
                  width={CELL - 1.5}
                  height={CELL - 1.5}
                  rx="1.5"
                  fill={
                    allowed
                      ? 'var(--accent)'
                      : j <= i
                        ? 'var(--bg-sunken)'
                        : 'transparent'
                  }
                  opacity={allowed ? 0.85 : 1}
                />
              )),
            )}
          </svg>
        </div>
      </div>

      <div>
        <h4 style={{ fontFamily: 'var(--serif)', fontSize: '1.05rem' }}>{def.label}</h4>
        <p
          className="viz-note"
          style={{ fontFamily: 'var(--mono)', fontSize: '0.75rem', margin: '0.2rem 0 0.6rem' }}
        >
          {def.who}
        </p>
        <p className="viz-note">{def.note}</p>

        <div className="readout">
          <div>
            <span>cells computed</span>
            <span className="em">{(density * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span>of full causal</span>
            <span>{(N * (N + 1)) / 2} cells</span>
          </div>
          <div>
            <span>saved</span>
            <span className="em">{((1 - density) * 100).toFixed(0)}%</span>
          </div>
        </div>

        <p className="viz-note" style={{ marginTop: '0.9rem' }}>
          Filled squares are computed; empty ones are skipped entirely. Everything above
          the diagonal is masked by causality in all six patterns — a token can never see
          its own future.
        </p>
      </div>
    </div>
  )
}
