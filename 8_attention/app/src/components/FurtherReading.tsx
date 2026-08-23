import { RESOURCE_GROUPS } from '../data/resources'

/**
 * A way out of the page.
 *
 * Grouped by what the reader wants next rather than by type, because "links about
 * attention" is not a useful category and "read this if you want to build one" is.
 *
 * Every entry states what it will not do for you as well as what it will. That is the
 * same rule the mechanism cards follow, and it matters more for recommendations than
 * anywhere else - a resource suggested without its limits costs someone an afternoon
 * before they discover it was the wrong one.
 */
export function FurtherReading() {
  return (
    <div>
      {RESOURCE_GROUPS.map((g) => (
        <div className="rgroup" key={g.intent}>
          <h3 className="rgroup-intent">{g.intent}</h3>
          <p className="rgroup-blurb">{g.blurb}</p>

          <div className="rlist">
            {g.items.map((r) => (
              <a
                className="rcard"
                key={r.url}
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                <span className="rcard-label">
                  {r.label} <span aria-hidden="true">↗</span>
                </span>
                <span className="rcard-what">{r.what}</span>
                <span className="rcard-limit">
                  <strong>Not for:</strong> {r.limit}
                </span>
              </a>
            ))}
          </div>
        </div>
      ))}

      <div className="panel" style={{ marginTop: '1.5rem' }}>
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
          And the papers themselves
        </h4>
        <p style={{ margin: '0 0 0.6rem', maxWidth: '68ch' }}>
          Every mechanism above links to its own paper, and most of the ones with a
          canonical implementation link to that too &mdash; look for <em>Go deeper</em> on
          the card. The <em>See also</em> chips follow lineage rather than similarity, so
          walking them from MQA through GQA and MLA to TPA reads as one problem being
          attacked four different ways rather than as four unrelated entries.
        </p>
        <p style={{ margin: 0, color: 'var(--ink-muted)', maxWidth: '68ch' }}>
          All of those links are re-checked by <code>tools/check_links.py</code>, which
          fails the build on a 404 and shrugs at a timeout.
        </p>
      </div>
    </div>
  )
}
