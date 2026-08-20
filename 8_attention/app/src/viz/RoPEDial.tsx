import { useMemo, useState } from 'react'
import { applyRope, dot, ropeRelativeCheck } from '../lib/attention'

const R = 78
const CX = 100
const CY = 100

/**
 * RoPE's defining property, computed rather than asserted.
 *
 * Two tokens are rotated by an angle proportional to their absolute positions. The dot
 * product between them depends only on the difference of those angles, so sliding both
 * along the sequence leaves the score unchanged. The readout below checks exactly that
 * at four wildly different absolute positions, and the numbers agree because the maths
 * makes them agree - not because they were written to.
 */
export function RoPEDial() {
  const [posA, setPosA] = useState(2)
  const [offset, setOffset] = useState(6)
  const [base, setBase] = useState(10000)

  const posB = posA + offset

  // A single 2D slice - one RoPE rotation pair - so the geometry is visible.
  const qVec = useMemo(() => [1, 0.35], [])
  const kVec = useMemo(() => [0.9, 0.5], [])

  // The fastest-rotating pair, where p = 0, so theta is simply the position.
  const angle = (pos: number) => pos / Math.pow(base, 0)

  const rotA = applyRope(qVec, posA, base)
  const rotB = applyRope(kVec, posB, base)
  const score = dot(rotA, rotB)

  const checks = ropeRelativeCheck(qVec, kVec, -offset, [2, 200, 8000, 60000], base)
  const spread = Math.max(...checks.map((c) => c.score)) - Math.min(...checks.map((c) => c.score))

  const pt = (v: number[]) => ({ x: CX + v[0] * R * 0.75, y: CY - v[1] * R * 0.75 })
  const a = pt(rotA)
  const b = pt(rotB)

  return (
    <div className="viz-wrap">
      <div>
        <div className="controls">
          <label className="field">
            query position: {posA}
            <input
              type="range"
              min={0}
              max={40}
              value={posA}
              onChange={(e) => setPosA(Number(e.target.value))}
            />
          </label>
          <label className="field">
            offset between them: {offset}
            <input
              type="range"
              min={0}
              max={12}
              value={offset}
              onChange={(e) => setOffset(Number(e.target.value))}
            />
          </label>
          <label className="field">
            RoPE base: {base.toLocaleString()}
            <input
              type="range"
              min={3}
              max={6}
              step={0.5}
              value={Math.log10(base)}
              onChange={(e) => setBase(Math.round(10 ** Number(e.target.value)))}
            />
          </label>
        </div>

        <svg width={200} height={200} viewBox="0 0 200 200" role="img" aria-label="RoPE rotation">
          <circle cx={CX} cy={CY} r={R} fill="none" stroke="var(--border)" strokeWidth="1" />
          <line x1={CX - R} y1={CY} x2={CX + R} y2={CY} stroke="var(--border)" strokeWidth="1" />
          <line x1={CX} y1={CY - R} x2={CX} y2={CY + R} stroke="var(--border)" strokeWidth="1" />

          <path
            d={`M ${CX} ${CY} L ${a.x} ${a.y}`}
            stroke="var(--accent)"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
          <circle cx={a.x} cy={a.y} r="4.5" fill="var(--accent)" />
          <text x={a.x + 8} y={a.y - 4} fontSize="11" fill="var(--accent)" fontFamily="var(--mono)">
            q@{posA}
          </text>

          <path
            d={`M ${CX} ${CY} L ${b.x} ${b.y}`}
            stroke="var(--buys)"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
          <circle cx={b.x} cy={b.y} r="4.5" fill="var(--buys)" />
          <text x={b.x + 8} y={b.y + 12} fontSize="11" fill="var(--buys)" fontFamily="var(--mono)">
            k@{posB}
          </text>
        </svg>
      </div>

      <div>
        <p className="viz-note">
          One RoPE pair, drawn as a 2D rotation. Move the query position and both vectors
          swing together — the <strong>angle between them never changes</strong>, because
          it depends only on the offset. That is the whole trick: relative position falls
          out of the geometry, with no extra tensor and no parameters.
        </p>

        <div className="readout">
          <div>
            <span>angle q</span>
            <span>{((angle(posA) * 180) / Math.PI).toFixed(1)}°</span>
          </div>
          <div>
            <span>angle k</span>
            <span>{((angle(posB) * 180) / Math.PI).toFixed(1)}°</span>
          </div>
          <div>
            <span>score q·k</span>
            <span className="em">{score.toFixed(4)}</span>
          </div>
        </div>

        <p className="viz-note" style={{ marginTop: '0.9rem' }}>
          Same offset ({offset}) checked at four absolute positions:
        </p>
        <div className="readout">
          {checks.map((c) => (
            <div key={c.position}>
              <span>position {c.position.toLocaleString()}</span>
              <span className="em">{c.score.toFixed(6)}</span>
            </div>
          ))}
          <div style={{ borderTop: '1px solid var(--border)', marginTop: '0.4rem', paddingTop: '0.4rem' }}>
            <span>spread</span>
            <span className="em">{spread.toExponential(1)}</span>
          </div>
        </div>

        <p className="viz-note" style={{ marginTop: '0.9rem' }}>
          The spread is floating-point noise, not a modelling effect. Raising the base
          slows every rotation down, which is what buys longer context — and is exactly
          the dial that Positional Interpolation, NTK-aware scaling and YaRN all turn, in
          increasingly careful ways.
        </p>
      </div>
    </div>
  )
}
