import { Term } from './Term'

/**
 * Why attention mattered.
 *
 * The page could tell you exactly what attention computes and exactly what it costs, and
 * still never say what was bought. A reader could finish the on-ramp knowing the
 * arithmetic and not know why this particular idea produced the current era.
 *
 * Placed after the on-ramp rather than before it on purpose: "any two words are one hop
 * apart" means nothing until you know what a hop is.
 *
 * Every date here was checked against the arXiv API, like every other date on this site.
 */

interface Claim {
  n: string
  title: string
  before: string
  after: string
  why: React.ReactNode
  cite?: { label: string; url: string }
}

const CLAIMS: Claim[] = [
  {
    n: '1',
    title: 'Any two words became one hop apart',
    before:
      'A recurrent network passed information along a chain. To connect word 1 to word 100 it took 100 sequential steps, and the gradient rarely survived the trip — which is why long sentences fell apart and why an entire sentence had to be squeezed into a single vector.',
    after:
      'Attention compares word 1 and word 100 directly, in one operation, and the distance between them costs nothing extra. Nothing has to survive a journey.',
    why: (
      <>
        This is the part that fixed quality. It is also the part that is easiest to
        undervalue now, because every model has it and nobody remembers the alternative.
      </>
    ),
    cite: {
      label: 'Sequence to Sequence Learning — arXiv 1409.3215, 10 Sep 2014',
      url: 'https://arxiv.org/abs/1409.3215',
    },
  },
  {
    n: '2',
    title: 'It could all happen at once',
    before:
      'A recurrent network cannot begin word t+1 until it has finished word t. That is a hard sequential dependency, and it leaves most of a GPU idle no matter how large the machine is.',
    after:
      'Every pair of words is compared in a single matrix multiply. There is no ordering constraint inside a layer at all, so the whole sequence is processed in parallel.',
    why: (
      <>
        <strong>This is the one that actually caused the current era.</strong> Not better
        accuracy per parameter &mdash; the ability to saturate hardware. A model you can
        train on a trillion tokens in weeks is a different kind of object from one you
        cannot, and that difference is what attention unlocked.
      </>
    ),
  },
  {
    n: '3',
    title: 'What to look at became learned, not designed',
    before:
      'A convolution reads a fixed neighbourhood chosen in advance by whoever designed the architecture. It cannot know that in this sentence, this word matters and its neighbour does not.',
    after:
      'Attention reads whatever matches, decided from the content at run time. The same trained network attends differently to two different sentences.',
    why: (
      <>
        This is why one architecture handles syntax, coreference and long-range topic
        without a separate mechanism for each &mdash; and why the same machinery works for
        images, audio and protein sequences, none of which it was designed for.
      </>
    ),
  },
  {
    n: '4',
    title: 'And it kept paying out, predictably',
    before:
      'Before this, buying a bigger model was a gamble. There was no reliable way to say what another ten times the compute would get you.',
    after:
      'Loss falls as a smooth power law in compute, data and parameters. You can forecast what a model will be worth before you build it.',
    why: (
      <>
        Predictability is what justified the spending. Nobody commits billions to a hope;
        they commit it to a curve. That curve is the mechanism by which a 2017 attention
        paper turned into an industry.
      </>
    ),
    cite: {
      label:
        'Scaling Laws — arXiv 2001.08361, 23 Jan 2020 · refined by Chinchilla, arXiv 2203.15556, 29 Mar 2022',
      url: 'https://arxiv.org/abs/2001.08361',
    },
  },
]

export function Breakthrough() {
  return (
    <div>
      <div className="claims">
        {CLAIMS.map((c) => (
          <div className="claim" key={c.n}>
            <span className="claim-n">{c.n}</span>
            <div className="claim-body">
              <h3>{c.title}</h3>

              <div className="claim-ba">
                <div className="ba-before">
                  <span className="ba-tag">before</span>
                  <p>{c.before}</p>
                </div>
                <div className="ba-after">
                  <span className="ba-tag">with attention</span>
                  <p>{c.after}</p>
                </div>
              </div>

              <p className="claim-why">{c.why}</p>

              {c.cite && (
                <p className="claim-cite">
                  <a href={c.cite.url} target="_blank" rel="noopener noreferrer">
                    {c.cite.label} ↗
                  </a>
                </p>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="plant" style={{ marginTop: '1.5rem' }}>
        <strong>And here is what it cost.</strong> Every one of those four wins comes from
        the same decision &mdash; let every word look at every word. That is also precisely
        why the work grows with the <Term k="quadratic">square</Term> of the length, and why
        the stored{' '}
        <Term k="kv-cache">keys and values</Term> grow with both length and users. The
        other fifty-one entries on this page are the field trying to keep these four
        properties while paying less for them. Not one of them gets all four back for free.
      </div>
    </div>
  )
}
