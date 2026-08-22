import { useMemo, useState } from 'react'
import { attentionCost, type CostModel } from '../lib/attention'
import { Formula } from '../components/Formula'

const LENGTHS = [2_048, 8_192, 32_768, 131_072, 524_288, 1_048_576]

const MODELS: { id: CostModel; label: string; colour: string; who: string }[] = [
  { id: 'quadratic', label: 'Full attention  O(n²)', colour: 'var(--costs)', who: 'Transformer' },
  { id: 'window', label: 'Sliding window  O(n·w)', colour: 'var(--warn)', who: 'Mistral, Longformer' },
  { id: 'sparse-topk', label: 'Top-k sparse  O(n·k + n)', colour: 'var(--accent)', who: 'NSA, DSA' },
  { id: 'linear', label: 'Linear  O(n)', colour: 'var(--buys)', who: 'DeltaNet, Mamba' },
]

/**
 * Cost shapes across context length, on a log scale.
 *
 * These are asymptotic terms only, with constant factors deliberately ignored - which is
 * why FlashAttention is absent. It changes the constant and the memory traffic while
 * leaving the quadratic term intact, so no curve of this kind can represent it. Saying
 * that plainly matters more than drawing an extra line, because "which complexity class"
 * and "which is actually faster" are different questions, and the field spent 2020-2022
 * learning that the hard way.
 */
export function ScalingCurves() {
  const [w, setW] = useState(4096)
  const [k, setK] = useState(2048)

  const W = 520
  const H = 280
  const PAD = 52

  const series = useMemo(
    () =>
      MODELS.map((m) => ({
        ...m,
        points: LENGTHS.map((n) => ({ n, cost: attentionCost(m.id, n, w, k) })),
      })),
    [w, k],
  )

  const allCosts = series.flatMap((s) => s.points.map((p) => p.cost))
  const minC = Math.min(...allCosts)
  const maxC = Math.max(...allCosts)

  const x = (n: number) =>
    PAD +
    ((Math.log2(n) - Math.log2(LENGTHS[0])) /
      (Math.log2(LENGTHS[LENGTHS.length - 1]) - Math.log2(LENGTHS[0]))) *
      (W - PAD * 1.4)

  const y = (c: number) =>
    H - PAD - ((Math.log10(c) - Math.log10(minC)) / (Math.log10(maxC) - Math.log10(minC))) * (H - PAD * 1.5)

  return (
    <div className="viz-wrap">
      <div>
        <div className="controls">
          <label className="field">
            window w: {w.toLocaleString()}
            <input
              type="range"
              min={8}
              max={14}
              value={Math.log2(w)}
              onChange={(e) => setW(2 ** Number(e.target.value))}
            />
          </label>
          <label className="field">
            top-k: {k.toLocaleString()}
            <input
              type="range"
              min={6}
              max={13}
              value={Math.log2(k)}
              onChange={(e) => setK(2 ** Number(e.target.value))}
            />
          </label>
        </div>

        <div className="scroll-x">
          <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Attention cost against context length">
            <line x1={PAD} y1={H - PAD} x2={W - PAD * 0.4} y2={H - PAD} stroke="var(--border)" />
            <line x1={PAD} y1={PAD * 0.5} x2={PAD} y2={H - PAD} stroke="var(--border)" />

            {LENGTHS.map((n) => (
              <g key={n}>
                <line x1={x(n)} y1={PAD * 0.5} x2={x(n)} y2={H - PAD} stroke="var(--border)" strokeDasharray="2 4" opacity="0.5" />
                <text x={x(n)} y={H - PAD + 16} fontSize="9.5" textAnchor="middle" fill="var(--ink-faint)" fontFamily="var(--mono)">
                  {n >= 1024 ? `${n / 1024}K` : n}
                </text>
              </g>
            ))}

            {series.map((s) => (
              <g key={s.id}>
                <path
                  d={s.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(p.n)} ${y(p.cost)}`).join(' ')}
                  stroke={s.colour}
                  strokeWidth="2.5"
                  fill="none"
                  strokeLinejoin="round"
                />
                {s.points.map((p) => (
                  <circle key={p.n} cx={x(p.n)} cy={y(p.cost)} r="3" fill={s.colour} />
                ))}
              </g>
            ))}

            <text x={PAD} y={PAD * 0.5 - 6} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
              relative cost (log)
            </text>
            <text x={W - PAD * 0.4} y={H - PAD + 30} fontSize="10" textAnchor="end" fill="var(--ink-faint)" fontFamily="var(--mono)">
              context length (log)
            </text>
          </svg>
        </div>
      </div>

      <div>
        {series.map((s) => {
          const at1M = s.points[s.points.length - 1].cost
          const quad = series[0].points[series[0].points.length - 1].cost
          return (
            <div key={s.id} style={{ marginBottom: '0.7rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span
                  style={{
                    width: '14px',
                    height: '3px',
                    background: s.colour,
                    borderRadius: '2px',
                    flexShrink: 0,
                  }}
                />
                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>{s.label}</span>
              </div>
              <div
                style={{
                  fontSize: '0.75rem',
                  color: 'var(--ink-faint)',
                  fontFamily: 'var(--mono)',
                  paddingLeft: '1.4rem',
                }}
              >
                {s.who} · at 1M: {(at1M / quad).toExponential(1)}× of full
              </div>
            </div>
          )
        })}

        <Formula id="cost" active={null} />

        <p className="viz-note" style={{ marginTop: '1rem' }}>
          At 2K these curves are almost indistinguishable, which is why a chatbot can
          simply use full attention. At 1M they differ by orders of magnitude, which is
          why an agent cannot.
        </p>

        <p className="viz-note" style={{ marginTop: '0.7rem' }}>
          <strong>FlashAttention is deliberately absent.</strong> It changes the constant
          factor and the memory traffic, not the complexity class, so it cannot be drawn
          here — and yet it beat most of the approximations in wall-clock terms. Complexity
          class and actual speed are different questions, and the field spent 2020–2022
          learning that.
        </p>
      </div>
    </div>
  )
}
