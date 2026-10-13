import ConfidenceBadge from './ConfidenceBadge';

const AGENT_NAMES = {
  'Image Interpretation': 'Image',
  'Question Understanding': 'Question',
  'Difficulty Routing': 'Router',
  'Image Context Filter': 'Filter',
  'Reasoning & Answer': 'Reasoning',
  'Verification': 'Verify',
};

function formatDuration(ms) {
  if (ms === undefined || ms === null) return null;
  return (ms / 1000).toFixed(1) + 's';
}

export default function CaseHeader({ result, baselineResult, workflowResult }) {
  if (!result) return null;

  const modality = result.image_analysis?.modality || '—';
  const questionType = result.question_analysis?.question_type?.replace(/_/g, ' ') || '—';
  const verificationStatus = result.verification_status || '—';

  const statusTagClass =
    verificationStatus === 'verified' ? 'tag-green' :
    verificationStatus === 'partially_verified' ? 'tag-amber' :
    'tag-red';

  // Agent pipeline timings
  const agentTimings = result.agent_trace?.map(step => {
    const name = AGENT_NAMES[step.agent] || step.agent;
    const duration = formatDuration(step.duration_ms);
    return { name, duration, raw: step.duration_ms };
  }).filter(item => item.raw !== undefined) || [];

  // System timings - using CSS variable colors for theme consistency
  const systems = [
    { name: 'Agentic', duration: formatDuration(result.total_duration_ms), colorVar: '--color-accent' },
    { name: 'Workflow', duration: formatDuration(workflowResult?.duration_ms), colorVar: '--color-warning' },
    { name: 'Baseline', duration: formatDuration(baselineResult?.duration_ms), colorVar: '--color-text-muted' },
  ].filter(s => s.duration);

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

      {/* System Timing Comparison */}
      {systems.length > 0 && (
        <div className="case-header-item timing-section">
          <span className="case-header-label">System Times</span>
          <div className="timing-badges">
            {systems.map((sys, i) => (
              <span
                key={i}
                className="timing-badge"
                style={{ color: `var(${sys.colorVar})`, borderColor: `var(${sys.colorVar})`, background: `var(${sys.colorVar})15` }}
              >
                {sys.name}: <strong>{sys.duration}</strong>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Agent Pipeline Breakdown */}
      {agentTimings.length > 0 && (
        <div className="case-header-item timing-section">
          <span className="case-header-label">Pipeline Steps</span>
          <div className="timing-badges">
            {agentTimings.map((item, i) => (
              <span key={i} className="timing-badge timing-agent">
                {item.name}: <strong>{item.duration}</strong>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
