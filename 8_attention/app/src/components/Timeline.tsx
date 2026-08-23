import { useEffect, useMemo, useState } from 'react'
import mechanismsRaw from '../data/mechanisms.json'
import erasRaw from '../data/eras.json'
import { CATEGORY_LABELS, type Era, type Mechanism } from '../data/types'
import { useLevel } from './LevelContext'
import { TimelineAxis } from './TimelineAxis'
import { AttentionMatrix } from '../viz/AttentionMatrix'
import { MaskPatterns } from '../viz/MaskPatterns'
import { RoPEDial } from '../viz/RoPEDial'
import { KVCacheBar } from '../viz/KVCacheBar'
import { LinearState } from '../viz/LinearState'
import { ScalingCurves } from '../viz/ScalingCurves'
import { MultiHead } from '../viz/MultiHead'
import { AttentionFlow } from '../viz/player/AttentionFlow'
import { DecodeLoop } from '../viz/player/DecodeLoop'
import { MaskMorph } from '../viz/player/MaskMorph'

const MECHANISMS = mechanismsRaw as Mechanism[]
const ERAS = erasRaw as Era[]

/**
 * Each mechanism already recorded which diagram explains it, in its `viz` field, and the
 * UI used to ignore that entirely - so reading about RoPE gave you prose while the RoPE
 * dial sat in a different section. This is the mapping that reconnects them.
 *
 * Components are rendered only when a reader asks for the diagram. Mounting six
 * interactive visualisations across fifty-two open cards would make the page crawl.
 */
const VIZ: Record<string, () => JSX.Element> = {
  matrix: AttentionMatrix,
  mask: MaskPatterns,
  rope: RoPEDial,
  kv: KVCacheBar,
  state: LinearState,
  curves: ScalingCurves,
  multihead: MultiHead,
  flow: AttentionFlow,
  decode: DecodeLoop,
  morph: MaskMorph,
}

/** The six a newcomer should read if they read nothing else. */
const ESSENTIALS = [
  'scaled-dot-product-attention',
  'multi-head-attention',
  'rope',
  'gqa',
  'gated-deltanet',
  'nsa',
]

function formatDate(iso: string): string {
  const [y, m, d] = iso.split('-')
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return `${d} ${months[Number(m) - 1]} ${y}`
}

function dateKindLabel(m: Mechanism): string {
  switch (m.date_kind) {
    case 'arxiv_v1':
      return `arXiv ${m.arxiv} v1`
    case 'release':
      return 'product release'
    case 'forum_post':
      return 'forum post - no paper'
  }
}

function MechanismDetail({ m, onSelect }: { m: Mechanism; onSelect: (id: string) => void }) {
  const { config } = useLevel()
  const [showViz, setShowViz] = useState(config.diagramOpen)
  const Viz = m.viz ? VIZ[m.viz] : undefined

  // A reader who changes level mid-page should see the new default expansion, not
  // whatever they happened to have open under the previous one.
  useEffect(() => setShowViz(config.diagramOpen), [config.diagramOpen])

  return (
    <div className="detail">
      <div className="block">
        <h4>The problem it answered</h4>
        <p>{m.problem}</p>
      </div>

      <div className="block">
        <h4>How it works{config.prose === 'plain' ? '' : ' (technical)'}</h4>
        <p>{config.prose === 'plain' ? m.plain : m.mechanism}</p>
      </div>

      {Viz && (
        <div className="block">
          <button className="chip" aria-expanded={showViz} onClick={() => setShowViz((v) => !v)}>
            {showViz ? 'Hide the diagram' : 'Show the diagram'}
          </button>
          {showViz && (
            <div className="card-viz">
              <Viz />
            </div>
          )}
        </div>
      )}

      <div className="tradeoffs">
        <div className="tradeoff buys">
          <h4>What it buys</h4>
          <ul>
            {m.buys.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        </div>
        <div className="tradeoff costs">
          <h4>What it costs</h4>
          <ul>
            {m.costs.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="pickwhen">
        <h4>When you would actually pick it</h4>
        <p>{m.pick_when}</p>
      </div>

      {config.equation && m.math && (
        <div className="block">
          <h4>The essential maths</h4>
          <pre className="math">{m.math}</pre>
          {m.math_note && <p className="math-note">{m.math_note}</p>}
        </div>
      )}

      {config.traps && m.confused_with && (
        <div className="trap">
          <h4 style={{ color: 'var(--warn)' }}>Easy to get wrong</h4>
          <p>{m.confused_with}</p>
        </div>
      )}

      {config.sources && (
      <div className="srcline">
        <span>{dateKindLabel(m)}</span>
        <span>·</span>
        <span>{m.authors}</span>
        <span>·</span>
        <a href={m.url} target="_blank" rel="noopener noreferrer">
          source ↗
        </a>
        <span>·</span>
        <a href={`#m-${m.id}`}>link to this ↗</a>
      </div>
      )}

      {m.reading && m.reading.length > 0 && (
        <div className="block">
          <h4>Go deeper</h4>
          <ul className="reading">
            {m.reading.map((r) => (
              <li key={r.url}>
                <a href={r.url} target="_blank" rel="noopener noreferrer">
                  {r.label} ↗
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      {m.see_also.length > 0 && (
        <div className="block">
          <h4>See also</h4>
          <div className="seealso">
            {m.see_also.map((id) => {
              const target = MECHANISMS.find((x) => x.id === id)
              if (!target) return null
              return (
                <button key={id} className="chip" onClick={() => onSelect(id)}>
                  {target.name}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {config.sources && m.date_evidence && (
        <p
          style={{
            fontSize: '0.8rem',
            color: 'var(--ink-faint)',
            marginTop: '0.6rem',
            maxWidth: '68ch',
          }}
        >
          <strong>Date evidence:</strong> {m.date_evidence}
        </p>
      )}
    </div>
  )
}

function MechanismRow({
  m,
  open,
  onToggle,
  onSelect,
}: {
  m: Mechanism
  open: boolean
  onToggle: () => void
  onSelect: (id: string) => void
}) {
  const { config } = useLevel()
  return (
    <div id={`m-${m.id}`} style={{ scrollMarginTop: '4.5rem' }}>
      <button className="mech-row" aria-expanded={open} onClick={onToggle}>
        <span className="date">{formatDate(m.date)}</span>
        <span className="name">{m.name}</span>
        <span className="badges">
          <span className="badge">{CATEGORY_LABELS[m.category]}</span>
          {m.covered_status === 'definitely_covered' && (
            <span className="badge taught">taught in class</span>
          )}
          {m.required_by_assignment && <span className="badge required">required</span>}
        </span>
        <span className="oneline">{config.prose === 'plain' ? m.plain : m.problem}</span>
      </button>
      {open && <MechanismDetail m={m} onSelect={onSelect} />}
    </div>
  )
}

type Path = 'all' | 'essentials' | 'required' | 'taught'

export function Timeline() {
  const { config } = useLevel()
  const [open, setOpen] = useState<string | null>('scaled-dot-product-attention')
  const [path, setPath] = useState<Path>(config.scope === 'essentials' ? 'essentials' : 'all')
  const [query, setQuery] = useState('')

  // The level sets the default scope, and switching level resets to it. The chips
  // below remain live at every level, so nothing is ever locked away - a beginner
  // who wants all fifty-two is one click from them.
  useEffect(() => {
    setPath(config.scope === 'essentials' ? 'essentials' : 'all')
  }, [config.scope])

  const sorted = useMemo(() => [...MECHANISMS].sort((a, b) => a.date.localeCompare(b.date)), [])

  // Deep links: a shared #m-<id> should open that card on a cold load, not just scroll
  // past a collapsed row.
  useEffect(() => {
    const hash = window.location.hash.replace('#m-', '')
    if (hash && MECHANISMS.some((m) => m.id === hash)) {
      setOpen(hash)
      requestAnimationFrame(() =>
        document.getElementById(`m-${hash}`)?.scrollIntoView({ block: 'center' }),
      )
    }
  }, [])

  const select = (id: string) => {
    setOpen(id)
    setPath('all')
    setQuery('')
    requestAnimationFrame(() => {
      document.getElementById(`m-${id}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
      history.replaceState(null, '', `#m-${id}`)
    })
  }

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    return sorted.filter((m) => {
      if (path === 'essentials' && !ESSENTIALS.includes(m.id)) return false
      if (path === 'required' && !m.required_by_assignment) return false
      if (path === 'taught' && m.covered_status !== 'definitely_covered') return false
      if (!q) return true
      return (
        m.name.toLowerCase().includes(q) ||
        m.plain.toLowerCase().includes(q) ||
        m.problem.toLowerCase().includes(q) ||
        (m.class_term ?? '').toLowerCase().includes(q)
      )
    })
  }, [sorted, path, query])

  return (
    <div>
      <TimelineAxis mechanisms={MECHANISMS} onSelect={select} selected={open} />

      <div className="controls" style={{ marginTop: '1.5rem' }}>
        <button className="chip" aria-pressed={path === 'all'} onClick={() => setPath('all')}>
          Everything ({sorted.length})
        </button>
        <button
          className="chip"
          aria-pressed={path === 'essentials'}
          onClick={() => setPath('essentials')}
        >
          New here — the 6 essentials
        </button>
        <button
          className="chip"
          aria-pressed={path === 'required'}
          onClick={() => setPath('required')}
        >
          Required by the assignment
        </button>
        <button className="chip" aria-pressed={path === 'taught'} onClick={() => setPath('taught')}>
          Taught in class
        </button>

        <input
          className="search"
          type="search"
          placeholder="Search 52 mechanisms…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search mechanisms"
        />
      </div>

      {visible.length === 0 && (
        <p className="viz-note">Nothing matches “{query}”. Try a mechanism name, or clear the search.</p>
      )}

      {ERAS.map((era) => {
        const items = visible.filter((m) => m.era === era.id)
        if (items.length === 0) return null
        return (
          <div className="era" key={era.id} id={`era-${era.id}`}>
            <div className="era-head">
              <span className="range">{era.range}</span>
              <h3>{era.title}</h3>
              <p className="bill">
                <b>The bill:</b> {era.bill}
              </p>
              <p className="story">{era.story}</p>
              {era.mind_change && (
                <div className="mindchange">
                  <b>The field changes its mind</b>
                  {era.mind_change}
                </div>
              )}
            </div>
            <div className="mech-list">
              {items.map((m) => (
                <MechanismRow
                  key={m.id}
                  m={m}
                  open={open === m.id}
                  onToggle={() => setOpen(open === m.id ? null : m.id)}
                  onSelect={select}
                />
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export { MECHANISMS, ERAS, VIZ, ESSENTIALS }
