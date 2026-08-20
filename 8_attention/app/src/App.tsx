import { useEffect, useState } from 'react'
import { MECHANISMS, Timeline } from './components/Timeline'
import { FactCheck } from './components/FactCheck'
import { Onramp } from './components/Onramp'
import { AttentionVsCache } from './components/AttentionVsCache'
import { SelfCheck } from './components/SelfCheck'
import { Term } from './components/Term'
import { useLevel, LevelPicker } from './components/LevelContext'
import { AttentionMatrix } from './viz/AttentionMatrix'
import { MaskPatterns } from './viz/MaskPatterns'
import { RoPEDial } from './viz/RoPEDial'
import { KVCacheBar } from './viz/KVCacheBar'
import { LinearState } from './viz/LinearState'
import { ScalingCurves } from './viz/ScalingCurves'
import { AttentionFlow } from './viz/player/AttentionFlow'
import { DecodeLoop } from './viz/player/DecodeLoop'
import { MaskMorph } from './viz/player/MaskMorph'

const SECTIONS = [
  { id: 'onramp', label: 'Start here' },
  { id: 'baseline', label: 'The matrix' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'positions', label: 'Position' },
  { id: 'memory', label: 'The KV bill' },
  { id: 'sparsity', label: 'Sparsity' },
  { id: 'state', label: 'State' },
  { id: 'cost', label: 'Cost' },
  { id: 'factcheck', label: 'Fact-check' },
  { id: 'next', label: "What's next" },
]

/**
 * Dark is the default, and index.html already carries data-theme="dark" on <html>.
 * That matters: setting it here instead would run after first paint and flash white
 * on a light OS before hydration. This hook only handles changing and remembering it.
 */
const THEME_KEY = 'attention-theme'

function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark')

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(THEME_KEY)
      if (stored === 'light' || stored === 'dark') setTheme(stored)
    } catch {
      // Blocked storage; dark stays.
    }
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      window.localStorage.setItem(THEME_KEY, theme)
    } catch {
      // Not remembering is not worth breaking the page over.
    }
  }, [theme])

  return { theme, setTheme }
}

function TopBar({ active }: { active: string }) {
  const { theme, setTheme } = useTheme()

  return (
    <nav className="topbar">
      <div className="shell topbar-inner">
        <span className="brand">Attention, in order</span>
        {SECTIONS.map((s) => (
          <a key={s.id} href={`#${s.id}`}>
            <button className="navbtn" aria-current={active === s.id}>
              {s.label}
            </button>
          </a>
        ))}

        <span className="topbar-sep" aria-hidden="true" />
        <LevelPicker compact />

        <button
          className="navbtn"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          aria-label="Toggle colour theme"
          title="Toggle colour theme"
        >
          {theme === 'dark' ? '☀' : '☾'}
        </button>
      </div>
    </nav>
  )
}

function Section({
  id,
  eyebrow,
  title,
  intro,
  children,
}: {
  id: string
  eyebrow: string
  title: string
  intro: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section className="band" id={id}>
      <div className="shell">
        <div className="section-head">
          <span className="eyebrow">{eyebrow}</span>
          <h2>{title}</h2>
          <p>{intro}</p>
        </div>
        {children}
      </div>
    </section>
  )
}

export default function App() {
  const { config } = useLevel()
  const [active, setActive] = useState('onramp')

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id)
      },
      { rootMargin: '-45% 0px -50% 0px' },
    )
    for (const s of SECTIONS) {
      const el = document.getElementById(s.id)
      if (el) observer.observe(el)
    }
    return () => observer.disconnect()
  }, [])

  const arxivDated = MECHANISMS.filter((m) => m.date_kind === 'arxiv_v1').length
  const taught = MECHANISMS.filter((m) => m.covered_status === 'definitely_covered').length

  return (
    <>
      <TopBar active={active} />

      <header className="hero">
        <div className="shell">
          <div className="prose">
            <h1>Attention, in the order it actually happened</h1>
            <p className="lede">
              Vanilla attention was never wrong. It was <strong>expensive</strong>. Every
              mechanism that follows is somebody looking at that bill and trying to pay
              less of it — and each one gives something up to do it. Laid out by launch
              date, you can watch the field change its mind: first it wants exactness, then
              memory, then length, then memory again.
            </p>
            <p className="lede" style={{ marginTop: '0.9rem', fontSize: '1rem' }}>
              No background assumed. It starts from what a word is even doing when it pays
              attention to another word.
            </p>
          </div>

          <LevelPicker />

          <div className="stat-row">
            <div className="stat">
              <span className="n">{MECHANISMS.length}</span>
              <span className="l">mechanisms</span>
            </div>
            <div className="stat">
              <span className="n">{arxivDated}</span>
              <span className="l">dates verified against arXiv</span>
            </div>
            <div className="stat">
              <span className="n">2014–2026</span>
              <span className="l">chronological</span>
            </div>
            <div className="stat">
              <span className="n">{taught}</span>
              <span className="l">covered in the session</span>
            </div>
          </div>
        </div>
      </header>

      <Section
        id="onramp"
        eyebrow="Start here · no background needed"
        title="What attention actually does"
        intro={
          <>
            Four steps, one sentence, one word at a time. By the end of it you will have
            met the bill that every mechanism on this page is trying to pay down — and
            you will have seen exactly where it comes from.
          </>
        }
      >
        <Onramp />

        <div className="scene-slot">
          <h3 className="scene-title">Watch it happen</h3>
          <p className="scene-intro">
            The same four steps, animated end to end. The moment worth catching is step
            4: the bars change size, but the total never moves off 100%.
          </p>
          <AttentionFlow />
        </div>

        {config.checks && <SelfCheck after="onramp" />}
      </Section>

      <Section
        id="baseline"
        eyebrow="June 2017 · the baseline"
        title="All seven words at once"
        intro={
          <>
            Same computation as step 2, now run for every word simultaneously — which is
            what the model actually does. Rows are questions, columns are labels, each row
            sums to 100%. The two toggles are worth playing with: they break the
            mechanism in the two ways it can be broken.
          </>
        }
      >
        <AttentionMatrix />
      </Section>

      <Section
        id="timeline"
        eyebrow="The main event"
        title="Every mechanism, by launch date"
        intro={
          <>
            Ordered by the date each one actually appeared — not by how it was taught, and
            not grouped by family. Each entry answers a problem that existed at that
            moment. Click any dot or row for what it buys, what it costs, and when you
            would actually pick it.
          </>
        }
      >
        <Timeline />
        {config.checks && <SelfCheck after="timeline" />}
      </Section>

      <Section
        id="positions"
        eyebrow="The first debt"
        title="Position: the price of looking everywhere at once"
        intro={
          <>
            Because every word compares against every word simultaneously, nothing in the
            mechanism knows what order they came in. Shuffle the sentence and you get the
            same answer. Order has to be added deliberately, and every positional scheme
            since 2017 is an argument about how.
          </>
        }
      >
        <RoPEDial />
      </Section>

      <Section
        id="memory"
        eyebrow="The bill that shaped everything after 2019"
        title="You can train it — but can you serve it?"
        intro={
          <>
            When a model writes word 5,001 it needs the keys and values of the previous
            5,000. Keeping them is the <Term k="kv-cache">KV cache</Term>, and it grows
            with length <em>and</em> with every simultaneous user. Move the sliders and
            watch a chatbot-scale number turn into a datacentre-scale one.
          </>
        }
      >
        <div className="scene-slot">
          <h3 className="scene-title">Watch the cache build, word by word</h3>
          <p className="scene-intro">
            The same sentence written four ways. Full attention keeps every block
            forever; a sliding window plateaus; sinks pin the first few; a linear state
            never grows at all. Four mechanisms, one moving picture.
          </p>
          <DecodeLoop />
        </div>

        <KVCacheBar />
        <div style={{ marginTop: '2rem' }}>
          <AttentionVsCache />
        </div>
        {config.checks && <SelfCheck after="memory" />}
      </Section>

      <Section
        id="sparsity"
        eyebrow="One idea, six rules"
        title="Sparse attention is a single move"
        intro={
          <>
            Stop letting every question read every label. That is the entire family. What
            separates six years of research is the <em>rule</em> for choosing — and
            specifically whether a human drew the pattern in advance or the model worked it
            out from the content.
          </>
        }
      >
        <div className="scene-slot">
          <h3 className="scene-title">One pattern becoming the next</h3>
          <p className="scene-intro">
            Six years of sparse-attention research, morphing through one another. They
            look like unrelated inventions until you see them deform, and then it is
            obvious they are one move with different rules for choosing.
          </p>
          <MaskMorph />
        </div>

        <MaskPatterns />
        {config.checks && <SelfCheck after="sparsity" />}
      </Section>

      <Section
        id="state"
        eyebrow="Why linear attention forgot"
        title="Adding is not the same as writing"
        intro={
          <>
            Remove the softmax and attention becomes a loop with one fixed-size{' '}
            <Term k="state">memory</Term>, so nothing grows. But the naive update only{' '}
            <em>adds</em>: if a slot holds 40 and the model wants 55, it becomes 95. The
            delta rule reads first and writes only the difference. That one change carried
            the whole revival.
          </>
        }
      >
        <LinearState />
        {config.checks && <SelfCheck after="state" />}
      </Section>

      <Section
        id="cost"
        eyebrow="Why the answer depends on scale"
        title="A 2K chatbot and a 1M agent are different problems"
        intro={
          <>
            A mechanism that is right for one and wrong for the other is not a bad
            mechanism. At 2,000 words these curves are nearly identical and full attention
            is simply correct. At a million they diverge by orders of magnitude.
          </>
        }
      >
        <ScalingCurves />
      </Section>

      <Section
        id="factcheck"
        eyebrow="The part that is easiest to get wrong"
        title="Naming and dating traps"
        intro={
          <>
            Every date here was checked against its primary source, because a confident
            sentence is not evidence. These are the specific places where confident
            sentences are wrong — including two that an AI agent is very likely to get
            wrong unprompted.
          </>
        }
      >
        <FactCheck />
      </Section>

      <Section
        id="next"
        eyebrow="Reading the trend"
        title="So what comes next?"
        intro={
          <>
            The point of a timeline is that it lets you extrapolate. Here is what the last
            three years actually show, and what it implies.
          </>
        }
      >
        <div className="prose">
          <div className="panel" style={{ marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
              1. Nobody ships a pure anything any more
            </h3>
            <p style={{ margin: 0, color: 'var(--ink-muted)' }}>
              MiniMax-01 kept full-attention layers at 7:1. Kimi Linear kept them at 3:1.
              HySparse kept 5 out of 49. Every attempt to replace attention outright has
              ended up reintroducing it as a minority layer type. The open question has
              moved from <em>which mechanism</em> to <em>what ratio, and which layers</em> —
              which is exactly what FlashMorph (June 2026) turns into a search problem
              instead of a guess.
            </p>
          </div>

          <div className="panel" style={{ marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
              2. Sparsity stopped being drawn and started being learned
            </h3>
            <p style={{ margin: 0, color: 'var(--ink-muted)' }}>
              From 2019 to 2023 the patterns were chosen by humans in advance: strided,
              local, global, random. From NSA onward the model decides, and the decision is
              trained in rather than bolted on. The remaining cost is the selector itself —
              and HySparse&rsquo;s answer, reusing a full-attention layer as an exact oracle
              instead of paying for a separate scorer, is the first real attack on that
              overhead.
            </p>
          </div>

          <div className="panel" style={{ marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
              3. The cache is being compressed, not just shared
            </h3>
            <p style={{ margin: 0, color: 'var(--ink-muted)' }}>
              MQA and GQA make heads share, which costs quality. MLA compresses into a
              latent, which does not. Kwai Summary Attention (April 2026) goes further and
              questions the target itself: perhaps a fixed-size memory loses too much and a
              full cache costs too much, and dividing it by a constant is the honest middle.
              Expect more of this — rejecting &ldquo;as small as possible&rdquo; as the goal.
            </p>
          </div>

          <div className="panel">
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>The falsifiable guess</h3>
            <p style={{ margin: '0 0 0.6rem', color: 'var(--ink-muted)' }}>
              The three surviving ideas — trainable sparsity, compressed caches, and hybrid
              layer schedules — are converging on one recipe. If the pattern holds, the next
              frontier model is a hybrid whose layer schedule is <em>searched rather than
              chosen</em>, whose cheap layers <em>share</em> selection and cache with a few
              expensive ones instead of each paying for their own scorer, and whose
              positional scheme is dropped or heavily rescaled after pretraining rather than
              carried through it.
            </p>
            <p style={{ margin: 0, color: 'var(--ink-muted)' }}>
              What would falsify it: a frontier model shipping pure linear attention with no
              full-attention layers at all, or a return to fixed hand-drawn sparsity because
              learned selectors turn out not to be worth their overhead.
            </p>
          </div>
        </div>
      </Section>

      <footer className="site">
        <div className="shell prose">
          <p>
            Every date is checked against its primary source by{' '}
            <code>tools/verify_dates.py</code>, which queries the arXiv API and fails the
            build on any mismatch. Coverage is enforced by{' '}
            <code>tools/check_completeness.py</code>. Sources and methodology are documented
            in the repository README.
          </p>
          <p>
            <a
              href="https://github.com/rahulni/Indic_LLM/tree/main/8_attention"
              target="_blank"
              rel="noopener noreferrer"
            >
              Source repository ↗
            </a>
          </p>
        </div>
      </footer>
    </>
  )
}
