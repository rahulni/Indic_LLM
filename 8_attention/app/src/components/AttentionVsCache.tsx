import { Term } from './Term'

/**
 * Attention versus the KV cache.
 *
 * This exists because a student asked it out loud in the session - which makes it a
 * real fork in understanding rather than one imagined by whoever wrote the page. The two
 * get conflated constantly, and until they are separated the rest of the timeline does
 * not parse: half these mechanisms reduce compute, half reduce memory, and a reader who
 * thinks those are the same thing cannot tell which problem any given entry solves.
 */
export function AttentionVsCache() {
  return (
    <div>
      <div className="vs">
        <div className="vs-col">
          <span className="vs-tag">the work</span>
          <h3>Attention</h3>
          <p>
            The <strong>computation</strong>. Every query is compared against every key,
            softmaxed into a budget, and used to blend values together. This is what costs
            you arithmetic, and it is the thing that grows with the square of the length.
          </p>
          <p className="vs-fix">
            Reduced by: sparse patterns, sliding windows, top-k,{' '}
            <Term k="kernel">better kernels</Term>, linear attention.
          </p>
        </div>

        <div className="vs-col">
          <span className="vs-tag">the storage</span>
          <h3>
            The <Term k="kv-cache">KV cache</Term>
          </h3>
          <p>
            The <strong>memory</strong>. When a model writes word 5,001, it needs the keys
            and values of the previous 5,000. Those were already computed once, so they are
            kept rather than recalculated. This costs you RAM, and it grows with length{' '}
            <em>and</em> with every simultaneous user.
          </p>
          <p className="vs-fix">
            Reduced by: MQA, GQA, MLA, sliding windows, paged memory, or having no cache at
            all (linear attention).
          </p>
        </div>
      </div>

      <div className="panel" style={{ marginTop: '1rem' }}>
        <h4
          style={{
            fontSize: '0.72rem',
            textTransform: 'uppercase',
            letterSpacing: '0.07em',
            color: 'var(--ink-faint)',
            fontFamily: 'var(--sans)',
            fontWeight: 600,
            marginBottom: '0.5rem',
          }}
        >
          Why the confusion is worth clearing up
        </h4>
        <p style={{ margin: '0 0 0.7rem', maxWidth: '68ch' }}>
          Because <strong>reducing one often does nothing for the other</strong>, and if
          you do not hold them apart you cannot tell what a mechanism actually bought.
          Top-k attention skips comparisons but still stores every key. Sliding windows
          happen to cut both. GQA cuts only memory and leaves the arithmetic untouched.
        </p>
        <p style={{ margin: 0, maxWidth: '68ch', color: 'var(--ink-muted)' }}>
          It is also why HySparse&rsquo;s 2026 claim is pointed: earlier sparse designs
          reduced computation <em>without saving cache</em>, because a layer that skips
          reading a key still had to keep it. Having the cheap layers borrow the cache of
          the layer above is what finally cuts both at once.
        </p>
      </div>

      <div className="panel" style={{ marginTop: '1rem' }}>
        <h4
          style={{
            fontSize: '0.72rem',
            textTransform: 'uppercase',
            letterSpacing: '0.07em',
            color: 'var(--ink-faint)',
            fontFamily: 'var(--sans)',
            fontWeight: 600,
            marginBottom: '0.5rem',
          }}
        >
          One more distinction that trips people up
        </h4>
        <p style={{ margin: 0, maxWidth: '68ch' }}>
          Reading the prompt (<Term k="prefill">prefill</Term>) and writing the answer (
          <Term k="decode">decode</Term>) are limited by different things. Prefill can
          process every word at once, so it is bound by raw computing power. Decode
          produces one word at a time and has to re-read the whole cache for each, so it is
          bound by memory <em>speed</em>. That is why MQA sped up generation so
          dramatically without changing the arithmetic much at all &mdash; and why BFLA,
          which only accelerates prefill, is honest about being half a solution.
        </p>
      </div>
    </div>
  )
}
