import ConfidenceBadge from './ConfidenceBadge';

export default function AnswerCard({ result }) {
  if (!result) return null;

  const verificationStatus = result.verification_status || 'unknown';
  const statusLabel = verificationStatus.replace(/_/g, ' ');
  const displayAnswer =
    verificationStatus === 'low_confidence' || verificationStatus === 'unsupported'
      ? (result.guarded_answer || result.answer)
      : result.answer;

  const statusTagClass =
    verificationStatus === 'verified' ? 'tag-green' :
    verificationStatus === 'partially_verified' ? 'tag-amber' :
    'tag-red';

  return (
    <div className="card animate-fade-in">
      <div className="card-header">
        <span className="card-title">💡 Answer</span>
        <span className={`tag ${statusTagClass}`} style={{ textTransform: 'capitalize' }}>
          {statusLabel}
        </span>
      </div>
      <div className="card-content">
        <p className="answer-text">{displayAnswer}</p>
      </div>
      <div style={{ marginTop: 'var(--space-3)', display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
        <ConfidenceBadge level={result.confidence} />
        {result.should_flag_for_review && (
          <span className="tag tag-red">review recommended</span>
        )}
      </div>
      {result.suggested_disclaimer && (
        <div className="disclaimer-bar" style={{ marginTop: 'var(--space-3)' }}>
          <span>⚠️</span>
          <span>{result.suggested_disclaimer}</span>
        </div>
      )}
    </div>
  );
}
