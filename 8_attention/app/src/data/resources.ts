/**
 * Where to go after this page.
 *
 * Grouped by what a reader wants next rather than by type or date, because "a list of
 * links about attention" is not useful and "read this if you want the maths" is.
 *
 * Each entry says what it is good for *and* what it is not. That is the same discipline
 * the mechanism cards use, and it matters more for recommendations than for anything
 * else: a resource recommended without its limits wastes the reader's afternoon.
 *
 * Every URL here was requested and returned 200 before being written down, and
 * `tools/check_links.py` re-checks them.
 */

export interface Resource {
  label: string
  url: string
  what: string
  /** What it will not do for you. Never omitted. */
  limit: string
}

export interface ResourceGroup {
  intent: string
  blurb: string
  items: Resource[]
}

export const RESOURCE_GROUPS: ResourceGroup[] = [
  {
    intent: 'I want the intuition first',
    blurb: 'Start here if the matrices are the part that keeps sliding off.',
    items: [
      {
        label: 'The Illustrated Transformer',
        url: 'https://jalammar.github.io/illustrated-transformer/',
        what: 'Jay Alammar draws every step of a Transformer forward pass. Still the clearest picture-led explanation anyone has written.',
        limit: 'Predates everything on this timeline after 2018 — it explains the baseline, not the efficiency work that followed.',
      },
      {
        label: 'Lilian Weng — Attention? Attention!',
        url: 'https://lilianweng.github.io/posts/2018-06-24-attention/',
        what: 'Traces attention from the RNN era through to self-attention, which is exactly the arc the first two entries on this timeline cover.',
        limit: 'A 2018 post with later edits; treat the taxonomy as historical rather than current.',
      },
    ],
  },
  {
    intent: 'I want to build one',
    blurb: 'Code you can read end to end, rather than a framework you configure.',
    items: [
      {
        label: 'The Annotated Transformer',
        url: 'https://nlp.seas.harvard.edu/annotated-transformer/',
        what: 'The 2017 paper reproduced line by line as runnable PyTorch, with the prose interleaved. The single best way to convince yourself the maths is real.',
        limit: 'Teaching code. It is not fast, and it implements the 2017 architecture rather than anything you would train today.',
      },
      {
        label: 'Hugging Face — attention interface',
        url: 'https://huggingface.co/docs/transformers/en/attention_interface',
        what: 'How to swap the attention implementation under an existing model, which is the quickest way to feel the difference between eager, SDPA and FlashAttention.',
        limit: 'Covers the plumbing, not the ideas.',
      },
      {
        label: 'flash-linear-attention',
        url: 'https://github.com/fla-org/flash-linear-attention',
        what: 'Triton implementations of most of the linear-attention family in one repository — GLA, DeltaNet, Gated DeltaNet, GSA. If you want to run the 2024–25 half of this timeline, it is here.',
        limit: 'Assumes you already know why you want these; it is kernels, not exposition.',
      },
      {
        label: 'vLLM',
        url: 'https://docs.vllm.ai/en/latest/',
        what: 'The serving stack where PagedAttention, GQA and sparse attention stop being papers and become throughput numbers.',
        limit: 'Inference only, and the internals move fast enough that blog posts about it go stale quickly.',
      },
    ],
  },
  {
    intent: 'I want the whole field at once',
    blurb: 'For when you need the map rather than a route.',
    items: [
      {
        label: 'Efficient Attention Mechanisms for LLMs: A Survey',
        url: 'https://arxiv.org/abs/2507.19595',
        what: 'The most complete survey of the efficiency work, covering both the algorithmic and the hardware-level side. Good for finding the things this page does not cover.',
        limit: 'A survey is a snapshot. This one is already behind the 2026 entries on this timeline, which is the normal fate of surveys in this area.',
      },
    ],
  },
  {
    intent: 'I want to follow what actually ships',
    blurb: 'Papers and production are different populations, and the gap is the interesting part.',
    items: [
      {
        label: 'Sebastian Raschka — The Big LLM Architecture Comparison',
        url: 'https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison',
        what: 'Reads the architecture out of open-weight model releases and compares them side by side. The best answer to "which of these is anyone actually using".',
        limit: 'Only covers open-weight models, so the closed frontier is inferred rather than observed.',
      },
    ],
  },
]
