export default function VerificationPanel({ result }) {
  if (!result) return null;

  const checks = result.checks || {};
  const issues = result.issues_found || [];
  const verificationNotes = result.verification_notes || '';
  const fallbackRecommendation = result.fallback_recommendation || 'none';
  const qualityWarning = Boolean(result.image_analysis?.quality_warning);
  const imageQuality = result.image_analysis?.image_quality || 'unknown';

  const checkItems = [
    { key: 'answer_matches_question_type', label: 'Answer matches question type' },
    { key: 'explanation_grounded_in_evidence', label: 'Explanation grounded in evidence' },
    { key: 'answer_strength_appropriate', label: 'Answer strength appropriate' },
    { key: 'no_contradictions', label: 'No contradictions detected' },
    { key: 'format_appropriate', label: 'Answer format appropriate' },
  ];

  // If we don't have detailed checks, show a simpler view
  const hasDetailedChecks = Object.keys(checks).length > 0;

  return (
    <div className="verification-panel animate-fade-in" style={{ animationDelay: '0.3s' }}>
      <div className="card-header">
        <span className="card-title">🛡️ Verification & Safety</span>
      </div>

      {hasDetailedChecks ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
          {checkItems.map((item) => {
            const passed = checks[item.key];
            if (passed === undefined) return null;
            return (
              <div key={item.key} className={`verification-check ${passed ? 'passed' : 'failed'}`}>
                <span className="check-icon">{passed ? '✓' : '✗'}</span>
                <span>{item.label}</span>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ padding: 'var(--space-2) 0' }}>
          <div className="verification-check" style={{ color: 'var(--color-text-secondary)' }}>
            <span className="check-icon">ℹ️</span>
            <span>Verification: {result.verification_status?.replace(/_/g, ' ') || 'pending'}</span>
          </div>
        </div>
      )}

      {verificationNotes && (
        <div style={{
          marginTop: 'var(--space-3)',
          padding: 'var(--space-3)',
          background: 'var(--color-bg-input)',
          borderRadius: 'var(--radius-sm)',
          fontSize: 'var(--font-size-sm)',
          color: 'var(--color-text-secondary)',
          lineHeight: 1.6,
        }}>
          {verificationNotes}
        </div>
      )}

      {qualityWarning && (
        <div className="tag tag-amber" style={{ marginTop: 'var(--space-3)', display: 'inline-flex' }}>
          poor image quality: {imageQuality.replace(/_/g, ' ')}
        </div>
      )}

      {(fallbackRecommendation !== 'none' || result.should_flag_for_review) && (
        <div style={{ marginTop: 'var(--space-3)', display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
          {fallbackRecommendation !== 'none' && (
            <span className="tag tag-amber">
              fallback: {fallbackRecommendation.replace(/_/g, ' ')}
            </span>
          )}
          {result.should_flag_for_review && (
            <span className="tag tag-red">review recommended</span>
          )}
        </div>
      )}

      {issues.length > 0 && (
        <div style={{ marginTop: 'var(--space-3)' }}>
          <div className="card-title" style={{ marginBottom: 'var(--space-2)', color: 'var(--color-warning)' }}>
            ⚠️ Issues Found
          </div>
          {issues.map((issue, i) => (
            <div key={i} className="failure-item">
              <span className="icon">•</span>
              <span>{issue}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
