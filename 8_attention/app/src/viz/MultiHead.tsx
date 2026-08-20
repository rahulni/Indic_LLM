import { useState } from 'react'
import { formatBytes } from '../lib/attention'

/**
 * How multi-head attention splits an embedding, and what that actually costs.
 *
 * MHA had no diagram, which left the reader meeting MQA, GQA and MLA as three fixes for
 * a cost they had never been shown.
 *
 * One correction worth stating plainly, because the folk version of this is wrong:
 * splitting into heads does NOT multiply the KV cache. The heads partition the embedding
 * rather than duplicating it, so heads x head_dim is always d_model and the cache is
 * 2 x layers x d_model x tokens whatever the head count. Splitting is free in arithmetic
 * *and* in memory.
 *
 * What the head count actually decides is how much MQA and GQA can claw back, since they
 * collapse the stored width down to one head's worth (or one group's). More heads means
 * a bigger prize for sharing them - which is the real reason head count belongs in this
 * story at all.
 */
export function MultiHead() {
  const [heads, setHeads] = useState(8)

  const modelDim = 512
  const headDim = modelDim / heads
  const layers = 32
  const seqLen = 32_768
  const bytes = 2
  const groupSize = 4

  const perWidth = (width: number) => 2 * layers * width * seqLen * bytes

  const mhaCache = perWidth(modelDim) // heads x headDim === modelDim, always
  const mqaCache = perWidth(headDim)
  const gqaGroups = Math.max(1, Math.floor(heads / groupSize))
  const gqaCache = perWidth(gqaGroups * headDim)

  const W = 560
  const H = 176
  const barW = W - 40
  const segW = barW / heads

  return (
    <div className="viz-wrap">
      <div>
        <div className="controls">
          <label className="field">
            heads: {heads} &times; {headDim} dimensions each
            <input
              type="range"
              min={1}
              max={16}
              step={1}
              value={heads}
              onChange={(e) => setHeads(Number(e.target.value))}
            />
          </label>
        </div>

        <div className="scroll-x">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="One embedding split across heads, each keeping its own keys and values"
          >
            <text x={20} y={14} fontSize="11" fill="var(--ink-faint)" fontFamily="var(--mono)">
              one word&rsquo;s embedding &mdash; {modelDim} numbers wide
            </text>
            <rect
              x={20}
              y={22}
              width={barW}
              height={24}
              rx="4"
              fill="var(--bg-sunken)"
              stroke="var(--border-strong)"
            />

            {Array.from({ length: heads }).map((_, i) => (
              <g key={i}>
                <line
                  x1={20 + i * segW}
                  y1={46}
                  x2={20 + i * segW + segW / 2}
                  y2={72}
                  stroke="var(--border-strong)"
                />
                <rect
                  x={20 + i * segW + 2}
                  y={72}
                  width={Math.max(segW - 4, 3)}
                  height={24}
                  rx="3"
                  fill="var(--accent)"
                  opacity={0.85}
                />
                {segW > 44 && (
                  <text
                    x={20 + i * segW + segW / 2}
                    y={88}
                    fontSize="10"
                    textAnchor="middle"
                    fill="var(--bg)"
                    fontFamily="var(--mono)"
                  >
                    h{i + 1}
                  </text>
                )}
                <rect
                  x={20 + i * segW + 2}
                  y={106}
                  width={Math.max(segW - 4, 3)}
                  height={14}
                  rx="2"
                  fill="var(--cat-memory)"
                  opacity={0.8}
                />
                <rect
                  x={20 + i * segW + 2}
                  y={124}
                  width={Math.max(segW - 4, 3)}
                  height={14}
                  rx="2"
                  fill="var(--cat-memory)"
                  opacity={0.5}
                />
              </g>
            ))}

            <text x={20} y={66} fontSize="11" fill="var(--ink-faint)" fontFamily="var(--mono)">
              split into {heads} head{heads > 1 ? 's' : ''} &mdash; the width is divided, not copied
            </text>
            <text x={20} y={154} fontSize="11" fill="var(--ink-faint)" fontFamily="var(--mono)">
              keys and values, stored per head &mdash; {heads} &times; {headDim} = {modelDim} either way
            </text>
          </svg>
        </div>
      </div>

      <div>
        <p className="viz-note">
          Each head works on its own slice, so one can follow grammar while another follows
          the topic. Drag the slider: the slices get narrower as they get more numerous.
        </p>

        <div className="readout">
          <div>
            <span>MHA stored width</span>
            <span className="em">
              {heads} &times; {headDim} = {modelDim}
            </span>
          </div>
          <div>
            <span>MHA cache, 32k, 1 user</span>
            <span className="em">{formatBytes(mhaCache)}</span>
          </div>
          <div>
            <span>GQA ({gqaGroups} group{gqaGroups > 1 ? 's' : ''})</span>
            <span className="em">{formatBytes(gqaCache)}</span>
          </div>
          <div>
            <span>MQA (1 shared set)</span>
            <span className="em">{formatBytes(mqaCache)}</span>
          </div>
        </div>

        <p className="viz-note" style={{ marginTop: '0.9rem' }}>
          <strong>Notice what does not move.</strong> The MHA cache stays at{' '}
          {formatBytes(mhaCache)} no matter how many heads you choose, because the heads
          divide the embedding rather than duplicating it. Splitting is free in arithmetic
          and in memory alike &mdash; which is worth saying, because &ldquo;multi&#8209;head
          multiplies your cache&rdquo; is a common and wrong summary.
        </p>

        <p className="viz-note" style={{ marginTop: '0.7rem' }}>
          What the head count really decides is <strong>how big the prize is for sharing
          them</strong>. MQA collapses all that stored width down to a single head&rsquo;s
          worth, so at {heads} head{heads > 1 ? 's' : ''} it saves {heads}&times;. Push the
          slider up and the saving grows with it. That is the pressure MQA, GQA and MLA are
          all responding to &mdash; and each answers it with a different trade.
        </p>
      </div>
    </div>
  )
}
