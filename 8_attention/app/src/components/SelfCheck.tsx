import { useState } from 'react'
import checks from '../data/checks.json'

interface Check {
  id: string
  after: string
  question: string
  answer: string
  hint: string
}

const CHECKS = checks as Check[]

/**
 * A check-yourself question, placed where understanding actually forks.
 *
 * The hint appears before the answer deliberately. Being nudged and then getting it is
 * worth more than being told, and a reader who peeks straight to the answer has lost
 * nothing they had.
 */
export function SelfCheck({ after }: { after: string }) {
  const check = CHECKS.find((c) => c.after === after)
  const [stage, setStage] = useState<'closed' | 'hint' | 'answer'>('closed')

  if (!check) return null

  return (
    <div className="selfcheck">
      <h4>Check yourself</h4>
      <p className="q">{check.question}</p>

      {stage === 'closed' && (
        <div className="controls" style={{ marginBottom: 0 }}>
          <button className="chip" onClick={() => setStage('hint')}>
            Give me a hint
          </button>
          <button className="chip" onClick={() => setStage('answer')}>
            Show the answer
          </button>
        </div>
      )}

      {stage === 'hint' && (
        <>
          <p className="hint">{check.hint}</p>
          <div className="controls" style={{ marginBottom: 0 }}>
            <button className="chip" onClick={() => setStage('answer')}>
              Show the answer
            </button>
          </div>
        </>
      )}

      {stage === 'answer' && <p className="a">{check.answer}</p>}
    </div>
  )
}

export { CHECKS }
