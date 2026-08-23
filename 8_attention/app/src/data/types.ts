export type CoveredStatus = 'definitely_covered' | 'probably_covered' | 'not_in_transcript'

export type Category =
  | 'core'
  | 'positional'
  | 'kv-cache'
  | 'sparse'
  | 'linear-recurrent'
  | 'long-context'
  | 'compression'
  | 'hybrid'
  | 'systems'

export interface Mechanism {
  id: string
  name: string
  date: string
  date_kind: 'arxiv_v1' | 'release' | 'forum_post'
  arxiv: string | null
  url: string
  title_check?: string | null
  date_evidence?: string
  authors: string
  era: string
  category: Category
  required_by_assignment: boolean
  covered_status: CoveredStatus
  class_term: string | null
  /** Jargon-free explanation, shown by default. Enforced non-empty by check_completeness.py. */
  plain: string
  problem: string
  mechanism: string
  buys: string[]
  costs: string[]
  pick_when: string
  viz: string | null
  /** Core equation, shown only at the research level. Present on ~18 entries. */
  math?: string
  /** Provenance of the equation: verbatim from the paper, or the essential form. */
  math_note?: string
  confused_with: string | null
  /** Ids of related mechanisms, following lineage rather than similarity. */
  see_also: string[]
  /** Canonical implementation or writeup, where one exists. Checked by check_links.py. */
  reading?: { label: string; url: string }[]
}

export interface Era {
  id: string
  range: string
  title: string
  bill: string
  story: string
  mind_change: string | null
}

/**
 * Human labels for the category field. These exist because not everything the
 * assignment lists is an attention mechanism - sinusoidal encodings are a positional
 * scheme, YaRN is a rescaling trick, PagedAttention is a serving optimisation - and
 * saying so accurately is part of understanding them.
 */
export const CATEGORY_LABELS: Record<Category, string> = {
  core: 'attention mechanism',
  positional: 'positional scheme',
  'kv-cache': 'KV-cache optimisation',
  sparse: 'sparse attention',
  'linear-recurrent': 'linear / recurrent',
  'long-context': 'context extension',
  compression: 'memory compression',
  hybrid: 'hybrid architecture',
  systems: 'systems / kernels',
}
