import { useMemo } from 'react'
import type { Mechanism } from '../data/types'

/**
 * The timeline as an actual time axis.
 *
 * The brief's central claim is that you can see the field change its mind once the
 * mechanisms are laid out chronologically - and that claim is not testable against a
 * list grouped under era headings, which is what this page had before. Here each dot sits
 * at its real date, so the shape of the field's attention becomes visible directly: the
 * near-empty stretch after 2017, the pile-up in 2023 when models met users, and the
 * three-and-a-half year gap between MQA being published and anyone caring.
 *
 * Colour groups nine categories into five families. Nine categorical hues is past the
 * point where a reader can hold them apart; five is legible, and the coarser grouping is
 * the more interesting signal anyway - you can watch memory work cluster in 2023-24 and
 * sparsity take over in 2025.
 */

type Family = 'mechanism' | 'memory' | 'sparsity' | 'recurrence' | 'engineering'

const FAMILY_OF: Record<string, Family> = {
  core: 'mechanism',
  positional: 'mechanism',
  'kv-cache': 'memory',
  compression: 'memory',
  sparse: 'sparsity',
  'linear-recurrent': 'recurrence',
  hybrid: 'recurrence',
  systems: 'engineering',
  'long-context': 'engineering',
}

const FAMILY_LABEL: Record<Family, string> = {
  mechanism: 'the mechanism itself',
  memory: 'memory & KV cache',
  sparsity: 'sparsity',
  recurrence: 'recurrence & linear',
  engineering: 'engineering & length',
}

const FAMILY_VAR: Record<Family, string> = {
  mechanism: 'var(--cat-mech)',
  memory: 'var(--cat-memory)',
  sparsity: 'var(--cat-sparse)',
  recurrence: 'var(--cat-recur)',
  engineering: 'var(--cat-eng)',
}

const START = Date.UTC(2014, 0, 1)
const END = Date.UTC(2026, 11, 31)

const W = 1180
const PAD_L = 16
const PAD_R = 16
const LANE_H = 17
const DOT_R = 5.5
const MIN_GAP = 13 // px between dots before they are pushed to another lane

export function TimelineAxis({
  mechanisms,
  onSelect,
  selected,
  /**
   * The landing-page variant: shorter, quieter, no legend or annotations. The hero needs
   * to show what this site *is* within the first screen, and the shape of the dots does
   * that on its own - the explanatory furniture belongs with the full timeline below.
   */
  compact = false,
}: {
  mechanisms: Mechanism[]
  onSelect: (id: string) => void
  selected: string | null
  compact?: boolean
}) {
  const laneH = compact ? 10 : LANE_H
  const dotR = compact ? 3.6 : DOT_R
  const axisY = compact ? 132 : 232
  const x = (iso: string) => {
    const t = Date.parse(iso + 'T00:00:00Z')
    return PAD_L + ((t - START) / (END - START)) * (W - PAD_L - PAD_R)
  }

  // Beeswarm: walk in date order and drop each dot into the lowest lane whose previous
  // dot is far enough to the left. Dense periods grow upward, which is what makes the
  // 2023-2025 pile-up read as a pile-up.
  const placed = useMemo(() => {
    const laneLastX: number[] = []
    return [...mechanisms]
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((m) => {
        const px = x(m.date)
        let lane = laneLastX.findIndex((last) => px - last >= MIN_GAP)
        if (lane === -1) {
          lane = laneLastX.length
          laneLastX.push(px)
        } else {
          laneLastX[lane] = px
        }
        return { m, px, lane }
      })
  }, [mechanisms])

  const maxLane = Math.max(0, ...placed.map((p) => p.lane))
  const height = axisY + (compact ? 26 : 44)
  const laneY = (lane: number) => axisY - (compact ? 14 : 22) - lane * laneH

  const years = Array.from({ length: 13 }, (_, i) => 2014 + i)

  const mqa = placed.find((p) => p.m.id === 'mqa')
  const gqa = placed.find((p) => p.m.id === 'gqa')

  const families: Family[] = ['mechanism', 'memory', 'sparsity', 'recurrence', 'engineering']

  return (
    <div>
      <div className="scroll-x">
        <svg
          width={W}
          height={height}
          viewBox={`0 0 ${W} ${height}`}
          role="img"
          aria-label="Every mechanism plotted at its real launch date between 2014 and 2026"
        >
          {/* year gridlines */}
          {years.map((y) => {
            const px = x(`${y}-01-01`)
            return (
              <g key={y}>
                <line
                  x1={px}
                  y1={laneY(maxLane) - 14}
                  x2={px}
                  y2={axisY}
                  stroke="var(--border)"
                  strokeDasharray="2 5"
                />
                <text
                  x={px}
                  y={axisY + 18}
                  fontSize="11"
                  textAnchor="middle"
                  fill="var(--ink-faint)"
                  fontFamily="var(--mono)"
                >
                  {y}
                </text>
              </g>
            )
          })}

          <line x1={PAD_L} y1={axisY} x2={W - PAD_R} y2={axisY} stroke="var(--border-strong)" />

          {/* The gap that makes the point: published 2019, adopted 2023. */}
          {!compact && mqa && gqa && (
            <g>
              <line
                x1={mqa.px}
                y1={axisY + 30}
                x2={gqa.px}
                y2={axisY + 30}
                stroke="var(--warn)"
                strokeWidth="1.5"
              />
              <line x1={mqa.px} y1={axisY + 26} x2={mqa.px} y2={axisY + 34} stroke="var(--warn)" strokeWidth="1.5" />
              <line x1={gqa.px} y1={axisY + 26} x2={gqa.px} y2={axisY + 34} stroke="var(--warn)" strokeWidth="1.5" />
              <text
                x={(mqa.px + gqa.px) / 2}
                y={axisY + 42}
                fontSize="10.5"
                textAnchor="middle"
                fill="var(--warn)"
              >
                MQA published, then ignored for 3½ years
              </text>
            </g>
          )}

          {placed.map(({ m, px, lane }) => {
            const family = FAMILY_OF[m.category] ?? 'mechanism'
            const isSel = selected === m.id
            return (
              <g
                key={m.id}
                className="axis-dot"
                onClick={() => onSelect(m.id)}
                role="button"
                tabIndex={0}
                aria-label={`${m.name}, ${m.date}`}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    onSelect(m.id)
                  }
                }}
              >
                <circle
                  cx={px}
                  cy={laneY(lane)}
                  r={isSel ? dotR + 2.5 : dotR}
                  fill={FAMILY_VAR[family]}
                  stroke={isSel ? 'var(--ink)' : 'var(--bg)'}
                  strokeWidth={isSel ? 2 : 1}
                />
                <title>{`${m.name} — ${m.date}`}</title>
              </g>
            )
          })}
        </svg>
      </div>

      {!compact && (
      <div className="legend">
        {families.map((f) => (
          <span className="legend-item" key={f}>
            <span className="legend-dot" style={{ background: FAMILY_VAR[f] }} />
            {FAMILY_LABEL[f]}
          </span>
        ))}
      </div>
      )}

      {!compact && (
      <p className="viz-note" style={{ marginTop: '0.6rem' }}>
        Each dot is one mechanism at its real launch date; dots stack upward where dates
        crowd together. Click one to jump to it. The shape is the argument: almost nothing
        between 2015 and 2018, a wall of work in 2023 when models met real users, and
        sparsity taking over from 2025.
      </p>
      )}
    </div>
  )
}
