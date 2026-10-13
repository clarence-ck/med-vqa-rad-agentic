export default function RoutingCard({ result }) {
  if (!result) return null;
  const rd = result.routing_decision;
  if (!rd) return null;

  const difficulty = rd.difficulty || 'unknown';
  const verificationPolicy = rd.verification_policy || 'standard';
  const fallbackPolicy = rd.fallback_policy || 'none';
  const routingRationale = rd.routing_rationale || '';
  const qualityWarning = Boolean(result.image_analysis?.quality_warning);
  const imageQuality = result.image_analysis?.image_quality || 'unknown';

  const difficultyClass =
    difficulty === 'easy' ? 'tag-green' :
    difficulty === 'hard' ? 'tag-red' :
    'tag-amber';

  return (
    <div className="card animate-fade-in" style={{ animationDelay: '0.18s' }}>
      <div className="card-header">
        <span className="card-title">🗺️ Routing Decision</span>
        <span className={`tag ${difficultyClass}`} style={{ textTransform: 'capitalize' }}>{difficulty}</span>
      </div>
      <div className="card-content">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
          <span className={`tag ${verificationPolicy === 'strict' ? 'tag-amber' : 'tag-gray'}`}>
            {verificationPolicy} verification
          </span>
          <span className={`tag ${fallbackPolicy === 'flag_for_review' ? 'tag-red' : fallbackPolicy === 'add_disclaimer' ? 'tag-amber' : 'tag-gray'}`}>
            fallback: {fallbackPolicy.replace(/_/g, ' ')}
          </span>
          {qualityWarning && (
            <span className="tag tag-amber">
              poor image quality: {imageQuality.replace(/_/g, ' ')}
            </span>
          )}
        </div>
        {routingRationale && (
          <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.6, margin: 0 }}>
            {routingRationale}
          </p>
        )}
      </div>
    </div>
  );
}
