import { useMemo, useState } from 'react'

/**
 * The delta rule, using the exact example from the session.
 *
 * A slot in the recurrent state holds 40. The model now wants it to be 55. Plain linear
 * attention adds, so the slot becomes 95 - a value nobody asked for. The delta rule
 * reads first, computes 55 - 40 = 15, and writes only that, so the slot lands on 55.
 *
 * Running both rules over the same sequence of writes shows the real consequence:
 * the additive state climbs without limit and saturates, while the delta state tracks
 * the target. That saturation is why plain linear attention forgets, and why fixing it
 * mattered enough to carry the whole linear-attention revival.
 */

interface Write {
  target: number
  label: string
}

const WRITES: Write[] = [
  { target: 40, label: 'store 40' },
  { target: 55, label: 'now 55' },
  { target: 55, label: 'still 55' },
  { target: 20, label: 'now 20' },
  { target: 70, label: 'now 70' },
  { target: 70, label: 'still 70' },
  { target: 30, label: 'now 30' },
]

export function LinearState() {
  const [step, setStep] = useState(WRITES.length - 1)
  const [decay, setDecay] = useState(1)

  const { additive, delta } = useMemo(() => {
    const additive: number[] = []
    const delta: number[] = []
    let sA = 0
    let sD = 0
    for (const w of WRITES) {
      // Plain linear attention: the new contribution is added to whatever is there.
      sA = sA * decay + w.target
      // Delta rule: read the slot, write only the difference.
      sD = sD * decay + (w.target - sD * decay)
      additive.push(sA)
      delta.push(sD)
    }
    return { additive, delta }
  }, [decay])

  const W = 460
  const H = 210
  const PAD = 34
  const maxV = Math.max(...additive, ...WRITES.map((w) => w.target)) * 1.1

  const x = (i: number) => PAD + (i / (WRITES.length - 1)) * (W - PAD * 2)
  const y = (v: number) => H - PAD - (v / maxV) * (H - PAD * 2)

  const line = (vals: number[]) =>
    vals.slice(0, step + 1).map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ')

  const targetLine = WRITES.slice(0, step + 1)
    .map((w, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(w.target)}`)
    .join(' ')

  return (
    <div className="viz-wrap">
      <div>
        <div className="controls">
          <label className="field">
            step: {step + 1} / {WRITES.length} — “{WRITES[step].label}”
            <input
              type="range"
              min={0}
              max={WRITES.length - 1}
              value={step}
              onChange={(e) => setStep(Number(e.target.value))}
            />
          </label>
          <label className="field">
            forget gate: {decay.toFixed(2)}
            <input
              type="range"
              min={0.5}
              max={1}
              step={0.05}
              value={decay}
              onChange={(e) => setDecay(Number(e.target.value))}
            />
          </label>
        </div>

        <div className="scroll-x">
          <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="State value over writes">
            <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" />
            <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" />

            <path d={targetLine} stroke="var(--ink-faint)" strokeWidth="1.5" strokeDasharray="4 3" fill="none" />
            <path d={line(additive)} stroke="var(--costs)" strokeWidth="2.5" fill="none" strokeLinejoin="round" />
            <path d={line(delta)} stroke="var(--buys)" strokeWidth="2.5" fill="none" strokeLinejoin="round" />

            {WRITES.slice(0, step + 1).map((w, i) => (
              <g key={i}>
                <circle cx={x(i)} cy={y(additive[i])} r="3.5" fill="var(--costs)" />
                <circle cx={x(i)} cy={y(delta[i])} r="3.5" fill="var(--buys)" />
                <text x={x(i)} y={H - PAD + 15} fontSize="9" textAnchor="middle" fill="var(--ink-faint)" fontFamily="var(--mono)">
                  {w.target}
                </text>
              </g>
            ))}

            <text x={PAD} y={PAD - 12} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
              state value
            </text>
          </svg>
        </div>
      </div>

      <div>
        <div className="readout">
          <div>
            <span>model wants</span>
            <span className="em">{WRITES[step].target}</span>
          </div>
          <div>
            <span style={{ color: 'var(--costs)' }}>additive (plain linear)</span>
            <span className="em" style={{ color: 'var(--costs)' }}>
              {additive[step].toFixed(1)}
            </span>
          </div>
          <div>
            <span style={{ color: 'var(--buys)' }}>delta rule</span>
            <span className="em" style={{ color: 'var(--buys)' }}>
              {delta[step].toFixed(1)}
            </span>
          </div>
          <div>
            <span>additive error</span>
            <span className="em">
              {(additive[step] - WRITES[step].target).toFixed(1)}
            </span>
          </div>
        </div>

        <p className="viz-note" style={{ marginTop: '0.9rem' }}>
          The state slot holds 40. The model now wants 55.{' '}
          <strong>Adding gives 95</strong> — a value nothing asked for. The delta rule
          reads the slot first, computes 55 − 40 = 15, and writes only that, landing on
          55.
        </p>

        <p className="viz-note" style={{ marginTop: '0.7rem' }}>
          Over many writes the additive state climbs without bound and saturates, which
          is exactly why plain linear attention forgets. Pull the forget gate below 1 to
          see what gating adds: coarse decay stops the runaway, but only the delta rule
          makes a precise correction. Gated DeltaNet uses both, because they fix
          different halves of the problem.
        </p>
      </div>
    </div>
  )
}
