import { useMemo, useState } from 'react'
import { attention, entropy, TOKENS } from '../lib/attention'

const CELL = 44
const PAD_L = 78
const PAD_T = 74

function heat(v: number): string {
  // Mix the two heat tokens in sRGB. A perceptual ramp would be nicer, but this keeps
  // the component free of a colour library and reads correctly in both themes.
  const t = Math.max(0, Math.min(1, v))
  return `color-mix(in srgb, var(--heat-1) ${(t * 100).toFixed(1)}%, var(--heat-0))`
}

/**
 * The baseline: real scaled dot-product attention over a seven-token sentence.
 *
 * The two toggles are the point of the whole component. Turning off 1/sqrt(d) sharpens
 * every row toward one-hot, and the entropy readout quantifies it. Turning off the
 * learned Q/K projections makes each token's best match itself, so the diagonal lights
 * up - which is the concrete reason queries and keys are separate networks.
 */
export function AttentionMatrix() {
  const [scaled, setScaled] = useState(true)
  const [useProjections, setUseProjections] = useState(true)
  const [causal, setCausal] = useState(true)
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null)

  const result = useMemo(
    () => attention(undefined, { scaled, useProjections, causal }),
    [scaled, useProjections, causal],
  )

  const n = TOKENS.length
  const width = PAD_L + n * CELL + 16
  const height = PAD_T + n * CELL + 16

  const meanEntropy = result.entropies.reduce((a, b) => a + b, 0) / n
  const maxEntropy = Math.log2(n)

  const focus = hover ?? { i: n - 1, j: 0 }
  const focusRow = result.weights[focus.i]
  const peak = Math.max(...focusRow)
  const peakToken = TOKENS[focusRow.indexOf(peak)]

  return (
    <div className="viz-wrap">
      <div>
        <div className="controls">
          <button
            className="chip"
            aria-pressed={scaled}
            onClick={() => setScaled((v) => !v)}
          >
            ÷ √d scaling {scaled ? 'on' : 'off'}
          </button>
          <button
            className="chip"
            aria-pressed={useProjections}
            onClick={() => setUseProjections((v) => !v)}
          >
            learned Q/K {useProjections ? 'on' : 'off'}
          </button>
          <button
            className="chip"
            aria-pressed={causal}
            onClick={() => setCausal((v) => !v)}
          >
            causal mask {causal ? 'on' : 'off'}
          </button>
        </div>

        <div className="scroll-x">
          <svg
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label="Attention weight matrix. Rows are queries, columns are keys."
          >
            <text
              x={PAD_L}
              y={16}
              fontSize="11"
              fill="var(--ink-faint)"
              fontFamily="var(--mono)"
            >
              keys (attended to) →
            </text>

            {TOKENS.map((t, j) => (
              <text
                key={`col-${j}`}
                x={PAD_L + j * CELL + CELL / 2}
                y={PAD_T - 10}
                fontSize="12"
                textAnchor="start"
                fill="var(--ink-muted)"
                fontFamily="var(--mono)"
                transform={`rotate(-45 ${PAD_L + j * CELL + CELL / 2} ${PAD_T - 10})`}
              >
                {t}
              </text>
            ))}

            {TOKENS.map((t, i) => (
              <text
                key={`row-${i}`}
                x={PAD_L - 10}
                y={PAD_T + i * CELL + CELL / 2 + 4}
                fontSize="12"
                textAnchor="end"
                fill="var(--ink-muted)"
                fontFamily="var(--mono)"
              >
                {t}
              </text>
            ))}

            {result.weights.map((row, i) =>
              row.map((w, j) => {
                const masked = result.scores[i][j] === -Infinity
                const isFocus = hover && hover.i === i && hover.j === j
                return (
                  <g key={`${i}-${j}`}>
                    <rect
                      x={PAD_L + j * CELL}
                      y={PAD_T + i * CELL}
                      width={CELL - 2}
                      height={CELL - 2}
                      rx="3"
                      fill={masked ? 'var(--bg-sunken)' : heat(w)}
                      stroke={isFocus ? 'var(--ink)' : 'transparent'}
                      strokeWidth="2"
                      onMouseEnter={() => setHover({ i, j })}
                      onMouseLeave={() => setHover(null)}
                    />
                    {!masked && (
                      <text
                        x={PAD_L + j * CELL + (CELL - 2) / 2}
                        y={PAD_T + i * CELL + (CELL - 2) / 2 + 4}
                        fontSize="10"
                        textAnchor="middle"
                        fontFamily="var(--mono)"
                        fill={w > 0.5 ? 'var(--bg)' : 'var(--ink-muted)'}
                        pointerEvents="none"
                      >
                        {w < 0.005 ? '·' : w.toFixed(2)}
                      </text>
                    )}
                  </g>
                )
              }),
            )}
          </svg>
        </div>
      </div>

      <div>
        <p className="viz-note">
          Rows are queries, columns are keys. Each row is a softmax, so it sums to 1.
          Every number is computed live from the six-dimensional token vectors — nothing
          here is drawn by hand.
        </p>

        <div className="readout">
          <div>
            <span>scale factor</span>
            <span className="em">
              {scaled ? `1/√6 = ${result.scale.toFixed(3)}` : '1.000'}
            </span>
          </div>
          <div>
            <span>mean row entropy</span>
            <span className="em">
              {meanEntropy.toFixed(2)} / {maxEntropy.toFixed(2)} bits
            </span>
          </div>
          <div>
            <span>row “{TOKENS[focus.i]}” peak</span>
            <span className="em">
              {(peak * 100).toFixed(0)}% → {peakToken}
            </span>
          </div>
          {hover && (
            <div>
              <span>raw q·k</span>
              <span className="em">{result.rawScores[hover.i][hover.j].toFixed(2)}</span>
            </div>
          )}
        </div>

        <p className="viz-note" style={{ marginTop: '0.9rem' }}>
          {!scaled && (
            <>
              <strong>Scaling off:</strong> entropy has dropped to{' '}
              {meanEntropy.toFixed(2)} bits. Dot products grow with dimension, so an
              unscaled softmax saturates toward one-hot and gradients through it vanish.
              That is the entire reason for the √d.
            </>
          )}
          {scaled && !useProjections && (
            <>
              <strong>Projections off:</strong> Q and K are now the raw embedding, so
              every token's strongest match is itself and the diagonal dominates. Separate
              query and key networks exist precisely to break this.
            </>
          )}
          {scaled && useProjections && (
            <>
              With learned projections, “bank” attends to “reserve” and “india” rather
              than to itself — the model is resolving which sense of the word this is.
              Try switching the projections off.
            </>
          )}
        </p>
      </div>
    </div>
  )
}

/** Shared by the sparse-pattern view so both report entropy the same way. */
export { entropy }
