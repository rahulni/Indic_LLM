/**
 * Render the whole app to a string in Node, so a component that throws is caught by a
 * command rather than by a reader.
 *
 * A green `tsc` only proves the types line up; it says nothing about whether the
 * components actually run. This mounts every section at every proficiency level,
 * exercises the real attention maths, and asserts that content the page is supposed to
 * contain is present. Effects do not run under SSR, which is fine - the observers, deep
 * links, theme persistence and animation loop are not what breaks.
 *
 * Run via:  npm run smoke
 */
import { renderToString } from 'react-dom/server'
import App from './App'
import { LevelProvider, LEVELS, LEVEL_CONFIG, type Level } from './components/LevelContext'
import { GLOSSARY } from './components/Term'
import { CHECKS } from './components/SelfCheck'
import { VIZ, MECHANISMS } from './components/Timeline'
import { SelfAttentionArcs } from './viz/SelfAttentionArcs'
import { Breakthrough } from './components/Breakthrough'
import { FORMULAS } from './data/formulas'
import { AttentionFlow } from './viz/player/AttentionFlow'
import { DecodeLoop } from './viz/player/DecodeLoop'
import { MaskMorph } from './viz/player/MaskMorph'
import { ease, lerp } from './viz/player/useBeats'
import {
  attention,
  kvBytesPerUser,
  DEFAULT_KV,
  buildMask,
  maskDensity,
} from './lib/attention'

const failures: string[] = []

function check(label: string, condition: boolean, detail = '') {
  if (!condition) failures.push(`${label}${detail ? ` - ${detail}` : ''}`)
}

function render(level: Level): string {
  try {
    return renderToString(
      <LevelProvider initial={level}>
        <App />
      </LevelProvider>,
    )
  } catch (err) {
    failures.push(`App threw at level "${level}": ${(err as Error).message}`)
    return ''
  }
}

// 1. Every level renders, and produces a substantial page.
const pages: Record<Level, string> = {
  new: render('new'),
  building: render('building'),
  research: render('research'),
}

for (const l of LEVELS) {
  check(`level "${l.id}" renders`, pages[l.id].length > 20_000, `${pages[l.id].length} chars`)
}

const html = pages.building

// 2. Content that must be present, including the on-ramp's load-bearing lines.
for (const needle of [
  'Attention, in the order it actually happened',
  'What attention actually does',
  'Every word writes three things',
  'soft blend, not a lookup', // the mental model that prevents misreading sparsity
  'attention sinks', // the payoff planted in step 3 of the on-ramp
  '196', // the n-squared punchline
  'How much do you already know',
  'Watch the cache build',
  'DroPE',
  'Gated Slot Attention',
  'What it costs',
  'Check yourself',
  'So what comes next',
]) {
  check(`page contains "${needle}"`, html.includes(needle))
}

// 3. REGRESSION GUARD. Formulas were written, verified, and then rendered in exactly
//    one gated place, so the default view carried none of them. They are now bound to
//    the visuals and must appear at EVERY level - including the beginner one, where a
//    glossed formula next to a picture teaches rather than intimidates.
for (const l of LEVELS) {
  check(
    `level "${l.id}" shows the attention equation`,
    pages[l.id].includes('Attention(Q, K, V)'),
  )
  check(`level "${l.id}" shows a formula gloss`, pages[l.id].includes('formula-gloss'))
}

// The opening picture and the breakthrough section must both be on the page.
check('self-attention arcs render', html.includes('how strongly each word attends'))
check('breakthrough section present', html.includes('What attention actually bought'))
check('breakthrough cites scaling laws', html.includes('2001.08361'))

// Every formula must have at least one glossed part, or its highlight explains nothing.
for (const [id, def] of Object.entries(FORMULAS)) {
  check(`formula "${id}" has a glossed part`, def.parts.some((p) => p.gloss))
}

// 4. The levels must actually differ in composition, not just wording. If these ever
//    collapse to the same output, the level selector has silently become decorative.
check('research level shows equations', pages.research.includes('The essential maths'))
check('building level hides equations', !pages.building.includes('The essential maths'))
check('building level shows self-checks', pages.building.includes('Check yourself'))
check('research level hides self-checks', !pages.research.includes('Check yourself'))
check(
  'new level lists fewer mechanisms than building',
  pages.new.length < pages.building.length,
  `new=${pages.new.length} building=${pages.building.length}`,
)

// 5. Trade-offs must survive at every level. Hiding what a technique costs would be the
//    one thing the level system must never do.
for (const l of LEVELS) {
  check(`level "${l.id}" still shows costs`, pages[l.id].includes('What it costs'))
}

// 6. Every mechanism needs the plain-English line, since two levels render it.
for (const m of MECHANISMS) {
  check(`${m.id} has plain text`, Boolean(m.plain && m.plain.trim()))
}

// 7. Any equation present must be non-empty and carry its provenance note.
const withMath = MECHANISMS.filter((m) => m.math)
check('some equations exist', withMath.length > 10, `${withMath.length}`)
for (const m of withMath) {
  check(`${m.id} equation non-empty`, Boolean(m.math && m.math.trim()))
  check(`${m.id} equation states provenance`, Boolean(m.math_note && m.math_note.trim()))
}

// 8. Every `viz` value must map to a real component, or a card silently loses its diagram.
for (const m of MECHANISMS) {
  if (m.viz) check(`${m.id} viz "${m.viz}" resolves`, m.viz in VIZ)
}
for (const key of ['flow', 'decode', 'morph', 'multihead']) {
  check(`VIZ registry has "${key}"`, key in VIZ)
}

// 9. Each animated scene renders standalone without throwing.
for (const [name, Scene] of [
  ['AttentionFlow', AttentionFlow],
  ['DecodeLoop', DecodeLoop],
  ['MaskMorph', MaskMorph],
  ['SelfAttentionArcs', SelfAttentionArcs],
  ['Breakthrough', Breakthrough],
] as const) {
  try {
    const out = renderToString(
      <LevelProvider>
        <Scene />
      </LevelProvider>,
    )
    check(`${name} renders`, out.length > 400, `${out.length} chars`)
  } catch (err) {
    failures.push(`${name} threw: ${(err as Error).message}`)
  }
}

// 10. The interpolation helpers the scenes depend on.
check('ease(0) === 0', ease(0) === 0)
check('ease(1) === 1', ease(1) === 1)
check('ease is monotonic', ease(0.3) < ease(0.6))
check('lerp clamps low', lerp(10, 20, -5) === 10)
check('lerp clamps high', lerp(10, 20, 5) === 20)

// 11. Glossary keys used in the source must exist.
for (const k of [
  'query', 'key', 'value', 'softmax', 'quadratic',
  'kv-cache', 'state', 'kernel', 'prefill', 'decode',
]) {
  check(`glossary defines "${k}"`, k in GLOSSARY)
}

// 12. Self-checks complete.
check('five self-checks defined', CHECKS.length === 5, `got ${CHECKS.length}`)
for (const c of CHECKS) {
  check(`check ${c.id} complete`, Boolean(c.question && c.answer && c.hint))
}

// 13. Level config must cover every declared level.
for (const l of LEVELS) {
  check(`config for "${l.id}"`, l.id in LEVEL_CONFIG)
}

// 14. The maths is real: scaling must genuinely change the distribution.
const scaled = attention(undefined, { scaled: true })
const unscaled = attention(undefined, { scaled: false })
const meanH = (r: typeof scaled) => r.entropies.reduce((a, b) => a + b, 0) / r.entropies.length
check(
  'unscaled attention is more peaked than scaled',
  meanH(unscaled) < meanH(scaled),
  `scaled=${meanH(scaled).toFixed(3)} unscaled=${meanH(unscaled).toFixed(3)}`,
)

// 14. Every softmax row sums to 1 (or 0 if fully masked).
for (const [i, row] of scaled.weights.entries()) {
  const sum = row.reduce((a, b) => a + b, 0)
  check(`row ${i} softmax sums to 1`, Math.abs(sum - 1) < 1e-9, `sum=${sum}`)
}

// 15. Causality: nothing may attend to its own future.
for (let i = 0; i < scaled.weights.length; i++) {
  for (let j = i + 1; j < scaled.weights.length; j++) {
    check(`causal mask holds at (${i},${j})`, scaled.weights[i][j] === 0)
  }
}

// 16. KV cache ordering must match the story the page tells.
const mha = kvBytesPerUser('mha', DEFAULT_KV)
const gqa = kvBytesPerUser('gqa', DEFAULT_KV)
const mqa = kvBytesPerUser('mqa', DEFAULT_KV)
const mla = kvBytesPerUser('mla', DEFAULT_KV)
check('MHA > GQA > MQA', mha > gqa && gqa > mqa)
check('MLA beats GQA', mla < gqa, `mla=${mla} gqa=${gqa}`)

// 17. Sparse patterns must actually be sparser than full causal.
const full = maskDensity(buildMask(24, 'causal'))
const win = maskDensity(buildMask(24, 'window', { windowSize: 5 }))
check('sliding window is sparser than causal', win < full, `window=${win} causal=${full}`)

if (failures.length) {
  console.error(`SMOKE FAILED (${failures.length}):`)
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}

console.log(
  `OK - 3 levels render (${pages.new.length.toLocaleString()} / ` +
    `${pages.building.length.toLocaleString()} / ${pages.research.length.toLocaleString()} chars), ` +
    `${MECHANISMS.length} mechanisms, ${withMath.length} card equations, ` +
    `${Object.keys(FORMULAS).length} bound formulas, 5 scenes, maths verified.`,
)
