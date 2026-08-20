/**
 * Real attention arithmetic, computed in the browser.
 *
 * Every number the diagrams show comes from these functions. Nothing is pre-rendered or
 * hand-drawn, so when a control changes the maths genuinely re-runs. That matters
 * because the most useful thing a reader can do here is turn the 1/sqrt(d) scaling off
 * and watch the softmax collapse - which only teaches anything if the softmax is real.
 */

export type Matrix = number[][]
export type Vector = number[]

/* ---------------------------------------------------------------- primitives */

export function dot(a: Vector, b: Vector): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  const n = a.length
  const k = b.length
  const m = b[0].length
  const out: Matrix = Array.from({ length: n }, () => new Array(m).fill(0))
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i][p]
      if (aip === 0) continue
      for (let j = 0; j < m; j++) out[i][j] += aip * b[p][j]
    }
  }
  return out
}

/**
 * Numerically stable softmax: subtract the row max before exponentiating, which is the
 * same trick FlashAttention generalises into its online form.
 *
 * `-Infinity` entries (how masking is represented here) produce exactly 0, and a row
 * that is entirely masked returns all zeros rather than NaN.
 */
export function softmax(row: Vector): Vector {
  let max = -Infinity
  for (const v of row) if (v > max) max = v
  if (!isFinite(max)) return row.map(() => 0)

  const exps = row.map((v) => (v === -Infinity ? 0 : Math.exp(v - max)))
  const sum = exps.reduce((a, b) => a + b, 0)
  return sum === 0 ? exps.map(() => 0) : exps.map((e) => e / sum)
}

/** Shannon entropy in bits. Used to quantify how peaked an attention row is. */
export function entropy(probs: Vector): number {
  let h = 0
  for (const p of probs) if (p > 0) h -= p * Math.log2(p)
  return h
}

/* ------------------------------------------------------------- the toy input */

/**
 * A deliberately tiny, hand-built example so the numbers stay legible.
 *
 * The embedding dimensions are named semantic axes rather than random noise, so the
 * resulting attention pattern is interpretable: "bank" genuinely scores highly against
 * "reserve" and "india" because they share the institution and place axes. Real models
 * learn these axes; here they are written down so the arithmetic can be followed by eye.
 */
export const DIMENSIONS = [
  'determiner',
  'institution',
  'finance',
  'place',
  'action',
  'quantity',
] as const

export const TOKENS = ['the', 'reserve', 'bank', 'of', 'india', 'raised', 'rates'] as const

// rows = tokens, columns = DIMENSIONS
export const EMBEDDINGS: Matrix = [
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // the
  [0.1, 0.9, 0.6, 0.0, 0.0, 0.2], // reserve
  [0.1, 0.9, 1.0, 0.1, 0.0, 0.1], // bank
  [0.9, 0.0, 0.0, 0.1, 0.0, 0.0], // of
  [0.1, 0.4, 0.1, 1.0, 0.0, 0.0], // india
  [0.0, 0.0, 0.2, 0.0, 1.0, 0.3], // raised
  [0.1, 0.2, 0.8, 0.0, 0.1, 0.9], // rates
]

/**
 * Fixed "learned" projections.
 *
 * These stand in for trained weights. Wq and Wk are deliberately different: that is the
 * whole reason a Transformer has separate query and key networks. If you set both to the
 * identity (the `useProjections: false` path below) every vector's best match becomes
 * itself, the diagonal dominates, and attention degenerates - which is exactly the
 * failure the Q/K split exists to prevent, and is worth seeing rather than being told.
 */
const W_Q: Matrix = [
  [0.2, 0.0, 0.1, 0.0, 0.0, 0.0],
  [0.0, 0.3, 0.8, 0.4, 0.0, 0.1],
  [0.0, 0.7, 0.2, 0.1, 0.0, 0.6],
  [0.0, 0.5, 0.1, 0.2, 0.0, 0.0],
  [0.0, 0.0, 0.1, 0.0, 0.8, 0.2],
  [0.0, 0.2, 0.6, 0.0, 0.1, 0.3],
]

const W_K: Matrix = [
  [0.2, 0.0, 0.0, 0.1, 0.0, 0.0],
  [0.0, 0.9, 0.2, 0.6, 0.0, 0.1],
  [0.0, 0.3, 0.7, 0.1, 0.0, 0.4],
  [0.0, 0.6, 0.1, 0.9, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.9, 0.1],
  [0.0, 0.1, 0.5, 0.0, 0.2, 0.8],
]

const W_V: Matrix = [
  [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

const IDENTITY: Matrix = DIMENSIONS.map((_, i) =>
  DIMENSIONS.map((_, j) => (i === j ? 1 : 0)),
)

/* ------------------------------------------------------------ masks / patterns */

export type MaskKind =
  | 'full'
  | 'causal'
  | 'strided'
  | 'window'
  | 'window-sink'
  | 'topk'
  | 'block-compressed'

export interface MaskOptions {
  windowSize?: number
  stride?: number
  sinks?: number
  topK?: number
  blockSize?: number
  /** Scores are needed for content-dependent patterns such as top-k. */
  scores?: Matrix
}

/**
 * Build a boolean "may attend" matrix for a given sparsity pattern.
 *
 * Every pattern here is causal, because these are all decoder mechanisms. Presenting
 * them on one shared grid is the point: the sparse family is not many different ideas,
 * it is one idea - restrict which keys a query may see - with different rules for
 * choosing. Only top-k and block-compressed look at content; the rest are drawn in
 * advance by a human, which is precisely their weakness.
 */
export function buildMask(n: number, kind: MaskKind, opts: MaskOptions = {}): boolean[][] {
  const {
    windowSize = 3,
    stride = 3,
    sinks = 1,
    topK = 3,
    blockSize = 2,
    scores,
  } = opts

  const allow: boolean[][] = Array.from({ length: n }, () => new Array(n).fill(false))

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      switch (kind) {
        case 'full':
          allow[i][j] = true
          break
        case 'causal':
          allow[i][j] = true
          break
        case 'strided':
          // Local neighbours, plus every stride-th token so information can still
          // route globally in two hops.
          allow[i][j] = i - j < windowSize || j % stride === 0
          break
        case 'window':
          allow[i][j] = i - j < windowSize
          break
        case 'window-sink':
          // The sliding window, plus the first few tokens pinned permanently. Those
          // pinned tokens are the attention sinks: softmax has to put its mass
          // somewhere, and without them eviction forces it onto tokens nobody wanted.
          allow[i][j] = i - j < windowSize || j < sinks
          break
        case 'block-compressed': {
          // NSA-style: coarse block summaries everywhere, plus a fine local window.
          const sameBlock = Math.floor(j / blockSize) === Math.floor(i / blockSize)
          const blockRepresentative = j % blockSize === 0
          allow[i][j] = sameBlock || blockRepresentative || i - j < windowSize
          break
        }
        case 'topk':
          // Filled in below, since it depends on the scores rather than on position.
          allow[i][j] = true
          break
      }
    }
  }

  if (kind === 'topk' && scores) {
    for (let i = 0; i < n; i++) {
      const candidates: { j: number; s: number }[] = []
      for (let j = 0; j <= i; j++) candidates.push({ j, s: scores[i][j] })
      candidates.sort((a, b) => b.s - a.s)
      const keep = new Set(candidates.slice(0, topK).map((c) => c.j))
      for (let j = 0; j < n; j++) allow[i][j] = keep.has(j)
    }
  }

  return allow
}

/** How much of the full causal attention matrix a pattern actually computes. */
export function maskDensity(allow: boolean[][]): number {
  const n = allow.length
  let kept = 0
  const causalCells = (n * (n + 1)) / 2
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) if (allow[i][j]) kept++
  return kept / causalCells
}

/* ------------------------------------------------------------------ attention */

export interface AttentionOptions {
  /** Divide scores by sqrt(d). Turning this off is the single most instructive control. */
  scaled?: boolean
  /** Use distinct learned Q/K projections. Off means Q = K = the raw embedding. */
  useProjections?: boolean
  causal?: boolean
  mask?: boolean[][]
  /** Softmax temperature. YaRN adjusts an equivalent quantity when stretching RoPE. */
  temperature?: number
}

export interface AttentionResult {
  q: Matrix
  k: Matrix
  v: Matrix
  /** Raw dot products, before scaling or masking. */
  rawScores: Matrix
  /** After scaling, temperature and masking; the input to the softmax. */
  scores: Matrix
  weights: Matrix
  output: Matrix
  /** Per-row entropy in bits - how spread out each query's attention is. */
  entropies: number[]
  scale: number
}

export function attention(
  embeddings: Matrix = EMBEDDINGS,
  opts: AttentionOptions = {},
): AttentionResult {
  const {
    scaled = true,
    useProjections = true,
    causal = true,
    mask,
    temperature = 1,
  } = opts

  const wq = useProjections ? W_Q : IDENTITY
  const wk = useProjections ? W_K : IDENTITY
  const wv = useProjections ? W_V : IDENTITY

  const q = matmul(embeddings, wq)
  const k = matmul(embeddings, wk)
  const v = matmul(embeddings, wv)

  const n = embeddings.length
  const d = q[0].length
  const scale = scaled ? 1 / Math.sqrt(d) : 1

  const rawScores: Matrix = []
  const scores: Matrix = []

  for (let i = 0; i < n; i++) {
    const raw: Vector = []
    const row: Vector = []
    for (let j = 0; j < n; j++) {
      const s = dot(q[i], k[j])
      raw.push(s)

      const blockedByCausality = causal && j > i
      const blockedByMask = mask ? !mask[i][j] : false
      row.push(blockedByCausality || blockedByMask ? -Infinity : (s * scale) / temperature)
    }
    rawScores.push(raw)
    scores.push(row)
  }

  const weights = scores.map(softmax)
  const output = matmul(weights, v)
  const entropies = weights.map(entropy)

  return { q, k, v, rawScores, scores, weights, output, entropies, scale }
}

/* ------------------------------------------------------------------ positions */

/**
 * Apply RoPE to a vector at a given position.
 *
 * Dimensions are taken in pairs and each pair is rotated by position * theta_pair,
 * where the angle shrinks geometrically across pairs. Low-index pairs rotate fast
 * (fine local distance), high-index pairs rotate slowly (coarse long-range position) -
 * which is exactly the split NTK-aware scaling and YaRN exploit when they stretch some
 * frequencies and leave others alone.
 */
export function applyRope(vec: Vector, position: number, base = 10000): Vector {
  const out = [...vec]
  const d = vec.length
  for (let p = 0; p < Math.floor(d / 2); p++) {
    const theta = position / Math.pow(base, (2 * p) / d)
    const c = Math.cos(theta)
    const s = Math.sin(theta)
    const x = vec[2 * p]
    const y = vec[2 * p + 1]
    out[2 * p] = x * c - y * s
    out[2 * p + 1] = x * s + y * c
  }
  return out
}

/**
 * Demonstrate RoPE's defining property: the score between a query at position i and a
 * key at position j depends only on (i - j).
 *
 * Returns the score for a fixed offset at several absolute positions. If RoPE is
 * behaving, every value in the returned array is the same. Drifting values at large
 * positions are floating-point limits appearing, which is a real effect worth showing
 * rather than hiding.
 */
export function ropeRelativeCheck(
  qVec: Vector,
  kVec: Vector,
  offset: number,
  positions: number[],
  base = 10000,
): { position: number; score: number }[] {
  return positions.map((pos) => ({
    position: pos,
    score: dot(applyRope(qVec, pos, base), applyRope(kVec, pos - offset, base)),
  }))
}

/** ALiBi's bias: a per-head slope times the distance, subtracted from the score. */
export function alibiSlopes(heads: number): number[] {
  // The paper's geometric sequence: for h heads, ratio 2^(-8/h).
  const ratio = Math.pow(2, -8 / heads)
  return Array.from({ length: heads }, (_, i) => Math.pow(ratio, i + 1))
}

/* ------------------------------------------------------------------ KV cache */

export type CacheScheme = 'mha' | 'gqa' | 'mqa' | 'mla'

export interface KvConfig {
  layers: number
  heads: number
  headDim: number
  seqLen: number
  users: number
  bytesPerNumber: number
  /** Query heads per KV group. Only used by GQA. */
  groupSize: number
  /** MLA's compressed latent dimension per token, per layer. */
  latentDim: number
}

export const DEFAULT_KV: KvConfig = {
  layers: 32,
  heads: 32,
  headDim: 128,
  seqLen: 32_768,
  users: 8,
  bytesPerNumber: 2, // fp16/bf16
  groupSize: 8,
  latentDim: 512,
}

/** Number of KV heads actually stored under each scheme. */
export function kvHeads(scheme: CacheScheme, cfg: KvConfig): number {
  switch (scheme) {
    case 'mha':
      return cfg.heads
    case 'gqa':
      return Math.max(1, Math.floor(cfg.heads / cfg.groupSize))
    case 'mqa':
      return 1
    case 'mla':
      return 0 // MLA stores a latent vector instead of per-head K and V.
  }
}

/**
 * Bytes of KV cache for one user.
 *
 * The factor of 2 is K and V. MLA is the exception: it caches a single low-rank latent
 * per token per layer and reprojects per head at use time, so it has no factor of 2 and
 * no head count - which is why it wins by so much more than GQA does.
 */
export function kvBytesPerUser(scheme: CacheScheme, cfg: KvConfig): number {
  if (scheme === 'mla') {
    return cfg.layers * cfg.seqLen * cfg.latentDim * cfg.bytesPerNumber
  }
  return (
    2 * cfg.layers * kvHeads(scheme, cfg) * cfg.headDim * cfg.seqLen * cfg.bytesPerNumber
  )
}

export function kvBytesTotal(scheme: CacheScheme, cfg: KvConfig): number {
  return kvBytesPerUser(scheme, cfg) * cfg.users
}

export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  let v = bytes
  let u = 0
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024
    u++
  }
  return `${v < 10 ? v.toFixed(2) : v.toFixed(1)} ${units[u]}`
}

/* --------------------------------------------------------------- cost curves */

export type CostModel = 'quadratic' | 'window' | 'linear' | 'sparse-topk'

/**
 * Relative attention cost in arbitrary units, for comparing shapes across lengths.
 *
 * These are the asymptotic terms only. They deliberately ignore constant factors, which
 * is precisely why FlashAttention does not appear here: it changes the constant and the
 * memory traffic while leaving the quadratic term intact, and no curve of this kind can
 * show that.
 */
export function attentionCost(model: CostModel, n: number, w = 4096, k = 2048): number {
  switch (model) {
    case 'quadratic':
      return n * n
    case 'window':
      return n * Math.min(w, n)
    case 'linear':
      return n
    case 'sparse-topk':
      // Indexer scores everything cheaply, then attends over k. Still has an O(n) term,
      // which is the honest reason "sparse" is not the same as "free".
      return n * Math.min(k, n) + n
  }
}
