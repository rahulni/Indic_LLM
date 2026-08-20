/**
 * Naming and dating traps.
 *
 * The assignment asks for the dates to be right and explicitly invites corrections. This
 * page collects the places where a confident-sounding sentence is wrong, including two
 * that an AI agent is very likely to get wrong unprompted. It is written as a list of
 * traps rather than as a list of anyone's mistakes, because that is what makes it useful
 * to the next person.
 */

interface Trap {
  claim: string
  actual: string
  why: string
  severity: 'high' | 'medium'
  sources: { label: string; url: string }[]
}

const TRAPS: Trap[] = [
  {
    claim: '“DroPE” refers to the paper titled “…Rotary Position Embedding for Efficient…”.',
    actual:
      'Two different papers have near-identical names. DroPE — “Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings” (Sakana AI, arXiv 2512.12167, v1 13 Dec 2025) — is the LLM context-extension method: train with RoPE, remove it, recalibrate briefly, and extend. DRoPE — “Directional Rotary Position Embedding for Efficient Agent Interaction Modeling” (arXiv 2503.15029, March 2025) — is an autonomous-driving agent-trajectory paper with no connection to language models.',
    why: 'The names differ only in capitalisation, and the papers are nine months and one research field apart. An agent asked to “explain DroPE” will very plausibly fetch the driving paper and write a fluent, entirely wrong section. The mechanism described in the session — train with RoPE, then drop it for a large length extension — is unambiguously the Sakana one.',
    severity: 'high',
    sources: [
      { label: 'DroPE — arXiv 2512.12167', url: 'https://arxiv.org/abs/2512.12167' },
      { label: 'DRoPE — arXiv 2503.15029', url: 'https://arxiv.org/abs/2503.15029' },
    ],
  },
  {
    claim: 'There is a paper introducing NTK-aware scaled RoPE.',
    actual:
      'There is not. NTK-aware scaling originated as a Reddit post by u/bloc97 in r/LocalLLaMA in late June 2023, with Dynamic-NTK following from u/emozilla. It was never published as a standalone paper; it was later folded into the YaRN paper as a baseline.',
    why: 'Any citation giving an arXiv identifier for “the NTK-aware scaling paper” is citing something that does not exist. This is the single most likely date on the required list to be invented, precisely because every other entry has a paper and this one does not. The name also references Neural Tangent Kernel theory by analogy — it is not a result derived from it.',
    severity: 'high',
    sources: [
      {
        label: 'Original r/LocalLLaMA post',
        url: 'https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/',
      },
      {
        label: 'Corroborating dated artifact — HF TGI issue #512',
        url: 'https://github.com/huggingface/text-generation-inference/issues/512',
      },
    ],
  },
  {
    claim: 'GSA and GQA are the same thing, or GSA is a mis-hearing of GQA.',
    actual:
      'GSA is Gated Slot Attention (arXiv 2409.07146, v1 11 Sep 2024, NeurIPS 2024) — a linear-attention method with bounded memory slots and adaptive forgetting. GQA is Grouped-Query Attention (arXiv 2305.13245) — a KV-head-sharing scheme for ordinary softmax attention. Different categories entirely.',
    why: 'The acronyms differ by one letter and both get described as “efficient attention”, so they collapse together easily. They were listed adjacently in the session, which is the clue that they are two things rather than one. GSA is also absent from the assignment’s required list despite being taught.',
    severity: 'medium',
    sources: [
      { label: 'GSA — arXiv 2409.07146', url: 'https://arxiv.org/abs/2409.07146' },
      { label: 'GQA — arXiv 2305.13245', url: 'https://arxiv.org/abs/2305.13245' },
    ],
  },
  {
    claim: 'The Transformer paper is from 2018.',
    actual:
      '“Attention Is All You Need” was submitted to arXiv on 12 June 2017 and published at NeurIPS in December 2017. There is no 2018 version.',
    why: 'The confusion is understandable: BERT (Oct 2018) is when Transformers became inescapable, so 2018 feels like the start. But the mechanism itself is mid-2017, and getting this wrong shifts the entire timeline’s zero point by a year.',
    severity: 'medium',
    sources: [{ label: 'arXiv 1706.03762', url: 'https://arxiv.org/abs/1706.03762' }],
  },
  {
    claim: 'Mistral invented sliding-window attention.',
    actual:
      'Mistral 7B (arXiv 2310.06825, Oct 2023) productionised it in a frontier decoder-only LLM. The mechanism is older: Sparse Transformers used local windows in April 2019, and Longformer built its architecture around them in April 2020.',
    why: 'This is the general pattern of “launched” versus “mattered”. Attribution drifts toward whoever shipped it at scale rather than whoever published it, and a chronological timeline is exactly where that error shows up.',
    severity: 'medium',
    sources: [
      { label: 'Longformer — arXiv 2004.05150', url: 'https://arxiv.org/abs/2004.05150' },
      { label: 'Mistral 7B — arXiv 2310.06825', url: 'https://arxiv.org/abs/2310.06825' },
    ],
  },
  {
    claim: 'YaRN is from September 2023.',
    actual: 'The arXiv v1 is 31 August 2023.',
    why: 'Its identifier is 2309.00071, and the 2309 prefix makes September the obvious guess. arXiv identifiers are assigned around the announcement cycle, not the submission date, so reading the month off the identifier is unreliable — the same trap makes Titans (2501.00663) look like 2025 when its v1 is 31 December 2024.',
    severity: 'medium',
    sources: [{ label: 'arXiv 2309.00071', url: 'https://arxiv.org/abs/2309.00071' }],
  },
  {
    claim: 'MQA was introduced around the same time as GQA.',
    actual:
      'MQA is from 6 November 2019. GQA is from 22 May 2023 — three and a half years later.',
    why: 'They are always taught together as points on one curve, which compresses them in memory. The gap is the interesting part: MQA solved a problem nobody had yet, because in 2019 almost nobody was serving long contexts to many concurrent users. It sat unused until the KV-cache bill came due.',
    severity: 'medium',
    sources: [
      { label: 'MQA — arXiv 1911.02150', url: 'https://arxiv.org/abs/1911.02150' },
      { label: 'GQA — arXiv 2305.13245', url: 'https://arxiv.org/abs/2305.13245' },
    ],
  },
  {
    claim: 'The delta rule is a 2024 idea.',
    actual:
      'The delta rule as fast-weight programming is arXiv 2102.11174, 22 February 2021. What arrived in 2024 (arXiv 2406.06484, 10 June 2024) is the algorithm that parallelises it over sequence length, which is what made it trainable at scale.',
    why: 'Not really an error so much as a nuance worth stating precisely: the idea and the thing that made the idea usable are three years apart, and “DeltaNet, 2024” is right about the second and wrong about the first.',
    severity: 'medium',
    sources: [
      { label: 'Delta rule — arXiv 2102.11174', url: 'https://arxiv.org/abs/2102.11174' },
      { label: 'Parallel DeltaNet — arXiv 2406.06484', url: 'https://arxiv.org/abs/2406.06484' },
    ],
  },
  {
    claim: 'NSA and DSA are two names for DeepSeek’s sparse attention.',
    actual:
      'They are two different mechanisms from the same lab, seven months apart. NSA (Native Sparse Attention, arXiv 2502.11089, Feb 2025) selects coarse blocks using three branches — compressed, selected, sliding window. DSA (DeepSeek Sparse Attention, released 29 Sep 2025 with V3.2-Exp) selects individual tokens using a lightning indexer feeding sparse MLA.',
    why: 'Same organisation, same year, similar names, both called “DeepSeek sparse attention” in casual writing. The distinction matters because their granularity differs — blocks versus tokens — and DSA composes with MLA while NSA does not.',
    severity: 'medium',
    sources: [
      { label: 'NSA — arXiv 2502.11089', url: 'https://arxiv.org/abs/2502.11089' },
      { label: 'DSA — DeepSeek release notes', url: 'https://api-docs.deepseek.com/news/news250929/' },
    ],
  },
]

const UNRESOLVED = {
  term: '“MSA”',
  text: 'Named in the spoken assignment brief alongside DeltaNet, GSA and GQA. It is most likely NSA — the session covered DeepSeek block compression with a low-rank indexer at length, which is NSA and DSA. But MLA is a plausible alternative reading, and the transcript does not disambiguate. Rather than guess, this site covers NSA, DSA and MLA in full and flags the ambiguity. Guessing here is exactly the failure mode the assignment warns about.',
}

export function FactCheck() {
  return (
    <div>
      {TRAPS.map((t, i) => (
        <div
          className="panel"
          key={i}
          style={{
            marginBottom: '1rem',
            borderLeftWidth: '3px',
            borderLeftColor: t.severity === 'high' ? 'var(--costs)' : 'var(--warn)',
          }}
        >
          <h4
            style={{
              fontSize: '0.72rem',
              textTransform: 'uppercase',
              letterSpacing: '0.07em',
              color: t.severity === 'high' ? 'var(--costs)' : 'var(--warn)',
              fontFamily: 'var(--sans)',
              fontWeight: 600,
              marginBottom: '0.4rem',
            }}
          >
            {t.severity === 'high' ? 'High risk of getting this wrong' : 'Commonly misstated'}
          </h4>

          <p style={{ margin: '0 0 0.7rem', fontWeight: 600, maxWidth: '68ch' }}>{t.claim}</p>

          <p style={{ margin: '0 0 0.7rem', maxWidth: '68ch' }}>
            <strong style={{ color: 'var(--buys)' }}>Actually:</strong> {t.actual}
          </p>

          <p
            style={{
              margin: '0 0 0.8rem',
              color: 'var(--ink-muted)',
              fontSize: '0.92rem',
              maxWidth: '68ch',
            }}
          >
            {t.why}
          </p>

          <div className="srcline" style={{ marginTop: 0, paddingTop: '0.6rem' }}>
            {t.sources.map((s) => (
              <a key={s.url} href={s.url} target="_blank" rel="noopener noreferrer">
                {s.label} ↗
              </a>
            ))}
          </div>
        </div>
      ))}

      <div
        className="panel"
        style={{ borderLeftWidth: '3px', borderLeftColor: 'var(--ink-faint)' }}
      >
        <h4
          style={{
            fontSize: '0.72rem',
            textTransform: 'uppercase',
            letterSpacing: '0.07em',
            color: 'var(--ink-faint)',
            fontFamily: 'var(--sans)',
            fontWeight: 600,
            marginBottom: '0.4rem',
          }}
        >
          Left unresolved on purpose
        </h4>
        <p style={{ margin: '0 0 0.5rem', fontWeight: 600 }}>{UNRESOLVED.term}</p>
        <p style={{ margin: 0, maxWidth: '68ch', color: 'var(--ink-muted)' }}>
          {UNRESOLVED.text}
        </p>
      </div>
    </div>
  )
}
