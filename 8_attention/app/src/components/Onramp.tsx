import { useMemo, useState } from 'react'
import { attention, dot, TOKENS } from '../lib/attention'
import { Term } from './Term'
import { useLevel } from './LevelContext'
import { Formula } from './Formula'

/**
 * The on-ramp: what attention actually does, before any matrix appears.
 *
 * The page used to open on a 7x7 heatmap of softmax weights, which is step 5 of 5. A
 * reader who does not already know what a query is sees noise. This builds the idea one
 * token at a time and only reaches the grid at the very end, where the quadratic cost is
 * introduced as the punchline rather than as an assumption.
 *
 * Two things in the copy here are load-bearing:
 *
 *  - Attention is a *soft blend*, not a lookup. A reader carrying a dictionary or
 *    hash-map intuition will misread every sparse mechanism further down the page.
 *  - The softmax budget must total 100% even when a query wants nothing. That single
 *    fact is what makes attention sinks necessary in 2023, and it is planted here so
 *    that entry can pay it off.
 *
 * Causal masking is deliberately switched off in this section. Restricting each word to
 * looking backwards is a separate idea, and introducing it here would obscure the one
 * being taught. It is flagged in the copy and demonstrated by the toggle further down.
 */

// Plain-language stand-ins for what each projection is doing for this sentence. Real
// models learn these; they are spelled out here so the idea is concrete rather than
// abstract. The embeddings driving the arithmetic below are the real ones.
const QUESTIONS = [
  'which thing am I attached to?',
  'am I part of a name here?',
  'which sense of me is this - money, or a riverside?',
  'what am I joining together?',
  'am I a place, or part of a title?',
  'who did this, and to what?',
  'rates of what, exactly?',
]

const LABELS = [
  'I am a determiner, mostly ignorable',
  'I am part of an institution name',
  'I am a financial institution',
  'I link two parts of a name',
  'I am a country and part of a title',
  'I am a past-tense action',
  'I am a financial quantity',
]

const CONTENT = [
  'almost nothing',
  'formality, central banking',
  'the institution, not the riverbank',
  'the joining itself',
  'India, the country',
  'an increase happened',
  'interest rates',
]

/** A small n x n grid, used to make the squaring visible rather than asserted. */
function GridSquare({ n }: { n: number }) {
  const size = 112
  const cell = size / n
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      {Array.from({ length: n }).map((_, i) =>
        Array.from({ length: n }).map((__, j) => (
          <rect
            key={`${i}-${j}`}
            x={j * cell}
            y={i * cell}
            width={Math.max(cell - 1, 1)}
            height={Math.max(cell - 1, 1)}
            fill="var(--accent)"
            opacity={0.75}
          />
        )),
      )}
    </svg>
  )
}

export function Onramp() {
  const { config } = useLevel()
  const [focus, setFocus] = useState(2) // "bank" - the word whose sense is ambiguous

  const result = useMemo(() => attention(undefined, { causal: false }), [])

  const scores = TOKENS.map((_, j) => dot(result.q[focus], result.k[j]))
  const weights = result.weights[focus]
  const maxScore = Math.max(...scores.map(Math.abs))

  const ranked = TOKENS.map((t, j) => ({ t, j, w: weights[j] })).sort((a, b) => b.w - a.w)

  return (
    <div>
      <div className="controls">
        <span
          style={{ fontSize: '0.82rem', color: 'var(--ink-faint)', marginRight: '0.2rem' }}
        >
          Follow one word:
        </span>
        {TOKENS.map((t, i) => (
          <button
            key={t}
            className="chip"
            aria-pressed={focus === i}
            onClick={() => setFocus(i)}
          >
            {t}
          </button>
        ))}
      </div>

      {/* -------------------------------------------------------------- step 1 */}
      <div className="step">
        <span className="step-n">1</span>
        <div className="step-body">
          <h3>Every word writes three things</h3>
          <p>
            {config.analogy
              ? 'Picture the sentence as a room of people, each holding a card. Before anyone speaks, every person writes down three things.'
              : 'Each token is projected three ways before any comparison happens.'}
          </p>

          <div className="qkv">
            <div className="qkv-box q">
              <h4>
                <Term k="query">Query</Term> &mdash; the question I am asking
              </h4>
              <p>{QUESTIONS[focus]}</p>
            </div>
            <div className="qkv-box k">
              <h4>
                <Term k="key">Key</Term> &mdash; the label I wear
              </h4>
              <p>{LABELS[focus]}</p>
            </div>
            <div className="qkv-box v">
              <h4>
                <Term k="value">Value</Term> &mdash; what I pass on
              </h4>
              <p>{CONTENT[focus]}</p>
            </div>
          </div>

          <p className="aside">
            The label and the content are kept separate on purpose. A word can be easy to
            find without that being what it contributes: &ldquo;reserve&rdquo; is worth{' '}
            <em>finding</em> when you are working out which sense of &ldquo;bank&rdquo; is
            meant, but what it <em>contributes</em> is the financial reading, not the word
            itself.
          </p>

          <Formula id="qkv" active={null} />
        </div>
      </div>

      {/* -------------------------------------------------------------- step 2 */}
      <div className="step">
        <span className="step-n">2</span>
        <div className="step-body">
          <h3>Compare that one question against every label</h3>
          <p>
            &ldquo;{TOKENS[focus]}&rdquo; holds up its question and checks it against all
            seven labels in the room. The comparison is a multiplication, and a bigger
            number means a better match. These are the real numbers:
          </p>

          <div className="bars">
            {TOKENS.map((t, j) => (
              <div className="bar-row" key={t}>
                <span className="bar-label">{t}</span>
                <span className="bar-track">
                  <span
                    className="bar-fill"
                    style={{
                      width: `${Math.max(1, (Math.abs(scores[j]) / maxScore) * 100)}%`,
                      opacity: j === focus ? 0.4 : 1,
                    }}
                  />
                </span>
                <span className="bar-num">{scores[j].toFixed(2)}</span>
              </div>
            ))}
          </div>

          <p className="aside">
            Nothing has been selected yet. Every word gets a score, including the ones that
            turn out not to matter &mdash; this is a <strong>soft blend, not a lookup</strong>.
            No word is ever fetched exactly; every word contributes something, even if only
            a trace. Hold on to that, because changing it is exactly what every
            &ldquo;sparse&rdquo; mechanism further down is trying to do.
          </p>

          <Formula id="attention" active="qk" />
        </div>
      </div>

      {/* -------------------------------------------------------------- step 3 */}
      <div className="step">
        <span className="step-n">3</span>
        <div className="step-body">
          <h3>Turn the scores into a budget that adds up to 100%</h3>
          <p>
            Raw scores are awkward: they can be any size, even negative. The{' '}
            <Term k="softmax">softmax</Term> turns them into percentages that are all
            positive and total exactly 100%. Now &ldquo;{TOKENS[focus]}&rdquo; has a fixed
            budget of attention to spend.
          </p>

          <div className="budget">
            {ranked.map(({ t, j, w }) => (
              <span
                key={j}
                className="budget-seg"
                style={{ width: `${Math.max(w * 100, 0.6)}%` }}
                title={`${t}: ${(w * 100).toFixed(1)}%`}
              >
                {w > 0.1 && (
                  <span className="budget-label">
                    {t} {(w * 100).toFixed(0)}%
                  </span>
                )}
              </span>
            ))}
          </div>

          <p>
            The output for &ldquo;{TOKENS[focus]}&rdquo; is now those percentages applied
            to what each word passes on, all added together. It is mostly &ldquo;
            {ranked[0].t}&rdquo;
            {ranked[1] ? <> and &ldquo;{ranked[1].t}&rdquo;</> : null}, with a trace of
            everything else.
          </p>

          <div className="plant">
            <strong>Remember this bit.</strong> The budget has to add up to 100%{' '}
            <em>whether or not anything is worth reading</em>. A word with no real question
            to ask still has to spend its full 100% somewhere. That sounds like a technical
            footnote. It is the reason <strong>attention sinks</strong> had to be invented
            in 2023, and you will meet it again on the timeline below.
          </div>

          <Formula id="attention" active="softmax" />
        </div>
      </div>

      {/* -------------------------------------------------------------- step 4 */}
      <div className="step">
        <span className="step-n">4</span>
        <div className="step-body">
          <h3>Now do that for every word at once &mdash; and meet the bill</h3>
          <p>
            You have followed one word. Every other word is doing exactly the same thing at
            the same time. Seven words, each checking seven labels, is 49 comparisons.
          </p>

          <div className="grid-compare">
            <figure>
              <GridSquare n={7} />
              <figcaption>
                7 words &rarr; <strong>49</strong> comparisons
              </figcaption>
            </figure>
            <span className="grid-arrow" aria-hidden="true">
              &rarr;
            </span>
            <figure>
              <GridSquare n={14} />
              <figcaption>
                14 words &rarr; <strong>196</strong> comparisons
              </figcaption>
            </figure>
          </div>

          <p>
            Double the words and the work goes up four times, not two, because each new
            word adds both a new question to ask <em>and</em> a new label for every
            existing question to check. Both sides of the square grow at once. That is what{' '}
            <Term k="quadratic">quadratic</Term> cost means.
          </p>

          <div className="plant">
            <strong>This is the story of the rest of this page.</strong> At a few thousand
            words nobody cares. At a million words that square is unaffordable &mdash; and
            every mechanism invented after 2017 is somebody looking at this bill and trying
            to pay less of it. Not one of them gets that for free.
          </div>

          <p className="aside">
            One simplification above: when a model is <em>writing</em> text, each word may
            only look at the words before it, never ahead. That is a separate idea called
            causal masking, and you can switch it on and off in the next section.
          </p>

          <Formula id="attention" active={null} />
        </div>
      </div>
    </div>
  )
}
