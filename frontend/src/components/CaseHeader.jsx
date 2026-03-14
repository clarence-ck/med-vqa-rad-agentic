import ConfidenceBadge from './ConfidenceBadge';

export default function CaseHeader({ result }) {
  if (!result) return null;

  const modality = result.image_analysis?.modality || '—';
  const questionType = result.question_analysis?.question_type?.replace(/_/g, ' ') || '—';
  const verificationStatus = result.verification_status || '—';

  const statusTagClass =
    verificationStatus === 'verified' ? 'tag-green' :
    verificationStatus === 'partially_verified' ? 'tag-amber' :
    'tag-red';

  return (
    <div className="case-header animate-fade-in">
      <div className="case-header-item">
        <span className="case-header-label">Modality</span>
        <span className="tag tag-blue">{modality}</span>
      </div>
      <div className="case-header-item">
        <span className="case-header-label">Question Type</span>
        <span className="tag tag-blue" style={{ textTransform: 'capitalize' }}>{questionType}</span>
      </div>
      <div className="case-header-item">
        <span className="case-header-label">Status</span>
        <span className={`tag ${statusTagClass}`} style={{ textTransform: 'capitalize' }}>
          {verificationStatus.replace(/_/g, ' ')}
        </span>
      </div>
      <div className="case-header-item" style={{ marginLeft: 'auto' }}>
        <ConfidenceBadge level={result.confidence} />
      </div>
      {result.total_duration_ms && (
        <div className="case-header-item">
          <span className="case-header-label">Time</span>
          <span className="case-header-value">{(result.total_duration_ms / 1000).toFixed(1)}s</span>
        </div>
      )}
    </div>
  );
}
