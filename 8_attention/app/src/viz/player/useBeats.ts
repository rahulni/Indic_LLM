import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Timing for the animated scenes.
 *
 * A scene is a list of beats. This hook reports which beat is current and how far
 * through it we are (`t`, from 0 to 1), so a scene can interpolate rather than snap.
 *
 * Two things it deliberately guarantees:
 *
 *  - **Nothing autoplays.** Motion starts only when a reader presses play. A page that
 *    begins moving on its own is hostile to anyone reading the text around it.
 *  - **Reduced motion is honoured properly.** Under `prefers-reduced-motion: reduce` the
 *    hook reports `t = 1` always, so every scene renders its landed state and the step
 *    buttons still walk through the whole explanation. The content is never carried by
 *    the motion itself - the animation is an aid, not the medium.
 *
 * The rAF loop runs only while playing, so several scenes open at once do not compete
 * for frames.
 */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    setReduced(mq.matches)
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  return reduced
}

export interface Beats {
  beat: number
  /** Progress through the current beat, 0 to 1. Always 1 under reduced motion. */
  t: number
  playing: boolean
  reduced: boolean
  toggle: () => void
  step: (delta: number) => void
  goTo: (beat: number) => void
}

export function useBeats(count: number, msPerBeat = 1700): Beats {
  const [beat, setBeat] = useState(0)
  const [t, setT] = useState(1)
  const [playing, setPlaying] = useState(false)
  const reduced = usePrefersReducedMotion()

  const frame = useRef<number>()
  const started = useRef<number>(0)

  useEffect(() => {
    if (!playing || reduced) return

    started.current = performance.now()

    const tick = (now: number) => {
      const elapsed = now - started.current
      if (elapsed >= msPerBeat) {
        started.current = now
        setBeat((b) => (b + 1) % count)
        setT(0)
      } else {
        setT(elapsed / msPerBeat)
      }
      frame.current = requestAnimationFrame(tick)
    }

    frame.current = requestAnimationFrame(tick)
    return () => {
      if (frame.current) cancelAnimationFrame(frame.current)
    }
  }, [playing, reduced, count, msPerBeat])

  const toggle = useCallback(() => {
    setPlaying((p) => {
      if (p) setT(1) // pausing should land on the finished state, not mid-transition
      return !p
    })
  }, [])

  const step = useCallback(
    (delta: number) => {
      setPlaying(false)
      setT(1)
      setBeat((b) => (b + delta + count) % count)
    },
    [count],
  )

  const goTo = useCallback((b: number) => {
    setPlaying(false)
    setT(1)
    setBeat(b)
  }, [])

  return { beat, t: reduced ? 1 : t, playing, reduced, toggle, step, goTo }
}

/** Ease-out, so a value settles rather than arriving at constant speed. */
export const ease = (x: number) => 1 - Math.pow(1 - x, 3)

/** Interpolate, clamped. */
export const lerp = (a: number, b: number, x: number) => a + (b - a) * Math.max(0, Math.min(1, x))
