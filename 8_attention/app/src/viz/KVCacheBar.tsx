import { useState } from 'react'
import { Formula } from '../components/Formula'
import {
  DEFAULT_KV,
  formatBytes,
  kvBytesPerUser,
  kvBytesTotal,
  kvHeads,
  type CacheScheme,
  type KvConfig,
} from '../lib/attention'

const SCHEMES: { id: CacheScheme; label: string; how: string }[] = [
  { id: 'mha', label: 'MHA', how: 'every head keeps its own K and V' },
  { id: 'gqa', label: 'GQA', how: 'heads share K/V within groups' },
  { id: 'mqa', label: 'MQA', how: 'one K/V for all heads' },
  { id: 'mla', label: 'MLA', how: 'K/V compressed into one latent vector' },
]

/**
 * The "you can train it, but can you serve it" calculator.
 *
 * The KV cache is the bill that drives most of this timeline, and it is far more
 * visceral as a number than as a paragraph. Every figure is computed from the config
 * below, and the arithmetic is printed so it can be checked rather than trusted.
 */
export function KVCacheBar() {
  const [cfg, setCfg] = useState<KvConfig>(DEFAULT_KV)

  const set = <K extends keyof KvConfig>(key: K, value: KvConfig[K]) =>
    setCfg((c) => ({ ...c, [key]: value }))

  const rows = SCHEMES.map((s) => ({
    ...s,
    heads: kvHeads(s.id, cfg),
    perUser: kvBytesPerUser(s.id, cfg),
    total: kvBytesTotal(s.id, cfg),
  }))

  const worst = Math.max(...rows.map((r) => r.total))

  return (
    <div>
      <div className="controls">
        <label className="field">
          context length: {cfg.seqLen.toLocaleString()} tokens
          <input
            type="range"
            min={10}
            max={20}
            value={Math.log2(cfg.seqLen)}
            onChange={(e) => set('seqLen', 2 ** Number(e.target.value))}
          />
        </label>
        <label className="field">
          concurrent users: {cfg.users}
          <input
            type="range"
            min={1}
            max={128}
            value={cfg.users}
            onChange={(e) => set('users', Number(e.target.value))}
          />
        </label>
        <label className="field">
          layers: {cfg.layers}
          <input
            type="range"
            min={8}
            max={96}
            step={4}
            value={cfg.layers}
            onChange={(e) => set('layers', Number(e.target.value))}
          />
        </label>
        <label className="field">
          query heads: {cfg.heads}
          <input
            type="range"
            min={8}
            max={64}
            step={8}
            value={cfg.heads}
            onChange={(e) => set('heads', Number(e.target.value))}
          />
        </label>
        <label className="field">
          GQA group size: {cfg.groupSize}
          <input
            type="range"
            min={1}
            max={16}
            value={cfg.groupSize}
            onChange={(e) => set('groupSize', Number(e.target.value))}
          />
        </label>
      </div>

      <div className="scroll-x">
        <table className="data">
          <thead>
            <tr>
              <th>scheme</th>
              <th>how</th>
              <th className="num">KV heads</th>
              <th className="num">per user</th>
              <th className="num">× {cfg.users} users</th>
              <th style={{ width: '26%' }}>relative</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>
                  <strong>{r.label}</strong>
                </td>
                <td style={{ color: 'var(--ink-muted)', fontSize: '0.82rem' }}>{r.how}</td>
                <td className="num">{r.id === 'mla' ? `${cfg.latentDim}d latent` : r.heads}</td>
                <td className="num">{formatBytes(r.perUser)}</td>
                <td className="num">
                  <strong>{formatBytes(r.total)}</strong>
                </td>
                <td>
                  <div
                    style={{
                      height: '10px',
                      borderRadius: '5px',
                      background: 'var(--accent)',
                      width: `${Math.max(1.5, (r.total / worst) * 100)}%`,
                      opacity: 0.85,
                    }}
                  />
                  <span
                    style={{
                      fontSize: '0.72rem',
                      color: 'var(--ink-faint)',
                      fontFamily: 'var(--mono)',
                    }}
                  >
                    {((r.total / worst) * 100).toFixed(0)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Formula id="kvCache" active="nkv" />

      <div className="readout">
        <div>
          <span>MHA arithmetic</span>
          <span className="em">
            2 × {cfg.layers} layers × {cfg.heads} heads × {cfg.headDim} dim ×{' '}
            {cfg.seqLen.toLocaleString()} tokens × {cfg.bytesPerNumber} bytes
          </span>
        </div>
        <div>
          <span>MLA arithmetic</span>
          <span className="em">
            {cfg.layers} layers × {cfg.seqLen.toLocaleString()} tokens × {cfg.latentDim}{' '}
            latent × {cfg.bytesPerNumber} bytes
          </span>
        </div>
      </div>

      <p className="viz-note" style={{ marginTop: '0.9rem', maxWidth: '68ch' }}>
        MLA has no factor of 2 and no head count because it caches a single low-rank
        latent per token per layer and reprojects each head's K and V at use time. That is
        why it wins by so much more than GQA, which only makes heads share. Push the
        context slider to 1M and watch the totals cross into terabytes — that is the
        constraint that produced everything in the 2024 and 2025 sections.
      </p>
    </div>
  )
}
