/**
 * Formulas that sit underneath the visuals, with the active term lit up.
 *
 * Every equation on this page used to live in one collapsed block inside a mechanism
 * card, gated to the research level - so on the default view the page contained no
 * formulas at all. These are the ones bound to a picture: the animation performs a term,
 * and the term highlights while it happens.
 *
 * A formula is a list of parts. Parts sharing a `key` highlight together, which is how an
 * opening bracket and its closing bracket light up as one thing. The gloss is taken from
 * the first part carrying that key, and it is what makes a formula readable by someone
 * who does not read formulas - without it these would make the beginner level worse.
 *
 * Provenance: the delta-rule and gated forms are verbatim from arXiv 2412.06464, fetched
 * rather than recalled. The scaled dot-product and multi-head forms are as published in
 * arXiv 1706.03762. The KV-cache and complexity expressions are definitional arithmetic -
 * they are what the calculators on this page actually evaluate.
 */

export interface FormulaPart {
  key: string
  text: string
  gloss?: string
}

export interface FormulaDef {
  parts: FormulaPart[]
  /** Shown under the formula when no part is active. */
  note?: string
}

export const FORMULAS: Record<string, FormulaDef> = {
  /** The master equation. Built up across the on-ramp, walked through by AttentionFlow. */
  attention: {
    parts: [
      {
        key: 'out',
        text: 'Attention(Q, K, V) = ',
        gloss:
          'The whole thing: for every word, a blend of what the other words carry, weighted by how well they match.',
      },
      {
        key: 'softmax',
        text: 'softmax(',
        gloss:
          'Turns raw scores into percentages that are all positive and add to exactly 100%. That fixed total is why attention sinks later had to be invented.',
      },
      {
        key: 'qk',
        text: ' Q K',
        gloss:
          'Every question checked against every label. This is the comparison step — and the reason cost grows with the square of the length, because it is every word against every word.',
      },
      { key: 'qk', text: 'ᵀ' },
      {
        key: 'scale',
        text: ' / √d',
        gloss:
          'Divide by the square root of the head width. Without it, dot products in high dimensions grow large, the softmax saturates towards one-hot, and no gradient survives.',
      },
      { key: 'scale', text: '_k' },
      { key: 'softmax', text: ' )' },
      {
        key: 'v',
        text: ' V',
        gloss:
          'The values — what each word actually passes on once someone decides to read it. Kept separate from the key, so a word can be easy to find without that being what it contributes.',
      },
    ],
    note: 'Vaswani et al., 2017 (arXiv 1706.03762), as published.',
  },

  /** What each of Q, K and V is. Used by on-ramp step 1, before the equation appears. */
  qkv: {
    parts: [
      { key: 'q', text: 'Q = X W', gloss: 'Query — what this word is looking for.' },
      { key: 'q', text: '_Q' },
      { key: 'sep1', text: '   ' },
      { key: 'k', text: 'K = X W', gloss: 'Key — the label this word advertises to others.' },
      { key: 'k', text: '_K' },
      { key: 'sep2', text: '   ' },
      { key: 'v', text: 'V = X W', gloss: 'Value — what this word passes on if it is read.' },
      { key: 'v', text: '_V' },
    ],
    note: 'Three different learned projections of the same word. Q and K are separate on purpose: if they were identical, every word’s best match would be itself.',
  },

  /** The bill that drives most of the timeline. */
  kvCache: {
    parts: [
      { key: 'lhs', text: 'cache bytes = ' },
      {
        key: 'two',
        text: '2',
        gloss: 'Two tensors per entry: one key and one value. MLA is the exception — it stores a single compressed latent instead, so it has no factor of 2.',
      },
      { key: 'layers', text: ' · L', gloss: 'Every layer keeps its own cache. Nothing is shared down the stack.' },
      {
        key: 'nkv',
        text: ' · n_kv',
        gloss:
          'How many key/value heads are actually stored. MHA keeps all of them, GQA one per group, MQA exactly one. This is the term MQA, GQA and MLA each attack differently.',
      },
      { key: 'dhead', text: ' · d_head', gloss: 'The width of each head.' },
      {
        key: 'tokens',
        text: ' · T',
        gloss:
          'The number of tokens held. This is the term that grows without limit — sliding windows cap it at the window size, and a linear state removes it entirely.',
      },
      { key: 'bytes', text: ' · b', gloss: 'Bytes per number — 2 for fp16, 1 for fp8.' },
      {
        key: 'users',
        text: ' · U',
        gloss:
          'And once more for every simultaneous user, because no two conversations can share a cache. This is why a model you can train can still be one you cannot serve.',
      },
    ],
    note: 'Definitional — this is the expression the calculator on this page evaluates.',
  },

  /** Plain linear attention: the update that only ever adds. */
  linearUpdate: {
    parts: [
      { key: 'lhs', text: 'S_t = ' },
      { key: 'carry', text: 'S_{t−1}', gloss: 'Whatever the fixed-size state already held.' },
      {
        key: 'add',
        text: ' + v_t k_tᵀ',
        gloss:
          'The new contribution is added on top. Nothing is ever removed, so a slot holding 40 that should become 55 becomes 95 instead — and the state silts up.',
      },
    ],
    note: 'Essential form, written to line up with the delta rule below.',
  },

  /** The delta rule: read before writing. Verbatim from arXiv 2412.06464. */
  deltaUpdate: {
    parts: [
      { key: 'lhs', text: 'S_t = ' },
      { key: 'carry', text: 'S_{t−1}' },
      {
        key: 'erase',
        text: '( I − β_t k_t k_tᵀ )',
        gloss:
          'Erase what this key already held before writing anything. This factor is the entire difference from the line above — the state is edited rather than accumulated.',
      },
      {
        key: 'write',
        text: ' + β_t v_t k_tᵀ',
        gloss: 'Then write the new value, at strength β_t.',
      },
    ],
    note: 'Verbatim from Gated Delta Networks (arXiv 2412.06464). Gated DeltaNet adds a decay gate α_t in front of the erase term.',
  },

  /** RoPE's defining property. */
  ropeIdentity: {
    parts: [
      { key: 'lhs', text: '⟨ R_m q , R_n k ⟩ = ' },
      {
        key: 'rhs',
        text: '⟨ q , R_{n−m} k ⟩',
        gloss:
          'Rotate a query at position m and a key at position n, and the result depends only on n − m. Relative position falls out of the geometry — no extra table, no extra parameters.',
      },
    ],
    note: 'The identity that makes RoPE work, in essential form (arXiv 2104.09864).',
  },

  /** Every sparse mechanism, as one expression with one thing varying. */
  sparseOutput: {
    parts: [
      { key: 'lhs', text: 'o_i = Σ' },
      {
        key: 'set',
        text: '_{ j ∈ 𝒮(i) }',
        gloss:
          'The allowed set — which keys word i is permitted to read. Every sparse mechanism on this page is a different rule for choosing 𝒮(i), and nothing else changes.',
      },
      { key: 'rest', text: '  softmax( q_i·k_j / √d ) v_j' },
    ],
    note: 'Full attention is 𝒮(i) = everything before i. That is the only difference.',
  },

  /** The three cost shapes. */
  cost: {
    parts: [
      {
        key: 'quadratic',
        text: 'O(n²)',
        gloss:
          'Full attention. Every word against every word, so doubling the length quadruples the work.',
      },
      { key: 'sep1', text: '     ' },
      {
        key: 'window',
        text: 'O(n·w)',
        gloss:
          'A sliding window of w. Linear in length once w is fixed — but anything outside the window is invisible in that layer.',
      },
      { key: 'sep2', text: '     ' },
      {
        key: 'linear',
        text: 'O(n)',
        gloss:
          'A fixed-size state. Memory is identical at a thousand tokens and a million — and no specific earlier word can be recovered from it exactly.',
      },
    ],
    note: 'Asymptotic terms only. FlashAttention is deliberately absent: it changes the constant and the memory traffic, not the class.',
  },
}

/** The gloss for a key, taken from the first part that carries one. */
export function glossFor(def: FormulaDef, key: string | null): string | undefined {
  if (!key) return undefined
  return def.parts.find((p) => p.key === key && p.gloss)?.gloss
}
