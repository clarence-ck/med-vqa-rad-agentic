import { useState } from 'react';

const STEP_META = {
  'Image Interpretation': { stage: 'Perception', subtitle: 'visual grounding', className: 'perception' },
  'Question Understanding': { stage: 'Perception', subtitle: 'intent + format', className: 'perception' },
  'Image Context Filter': { stage: 'Perception', subtitle: 'question-relevant findings', className: 'perception' },
  'Difficulty Routing': { stage: 'Planning', subtitle: 'difficulty + verification policy', className: 'routing' },
  'Reasoning & Answer': { stage: 'Reasoning', subtitle: 'final synthesis', className: 'reasoning' },
  'Verification': { stage: 'Safety', subtitle: 'support + confidence gate', className: 'verification' },
};

function getStepMeta(step) {
  return STEP_META[step.agent] || { stage: 'Stage', subtitle: '', className: 'default' };
}

export default function AgentTrace({ agentTrace, loading }) {
  const [expandedStep, setExpandedStep] = useState(null);

  const defaultSteps = [
    { agent: 'Image Interpretation', agent_id: 1 },
    { agent: 'Question Understanding', agent_id: 2 },
    { agent: 'Image Context Filter', agent_id: 3 },
    { agent: 'Difficulty Routing', agent_id: 4 },
    { agent: 'Reasoning & Answer', agent_id: 5 },
    { agent: 'Verification', agent_id: 6 },
  ];

  const steps = agentTrace || (loading ? defaultSteps : []);

  if (steps.length === 0) return null;

  const toggleStep = (id) => {
    setExpandedStep(expandedStep === id ? null : id);
  };

  const handleStepDoubleClick = (step) => {
    if (step.summary) {
      toggleStep(step.agent_id);
    }
  };

  return (
    <div className="card animate-fade-in" style={{ animationDelay: '0.2s' }}>
      <div className="card-header">
        <span className="card-title">🔗 Agent Pipeline</span>
        {agentTrace && (
          <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
            {agentTrace.length} steps
          </span>
        )}
      </div>

      {/* Timeline */}
      <div className="timeline" style={{ marginBottom: 'var(--space-3)' }}>
        {steps.map((step, i) => {
          const status = step.status || (loading ? 'pending' : 'pending');
          const dotClass = agentTrace ? (
            status === 'skipped' ? 'skipped' :
            status === 'pass' ? 'completed' :
            status === 'warning' ? 'completed' :
            status === 'fail' ? 'completed' : 'pending'
          ) : (loading ? (i === 0 ? 'active' : 'pending') : 'pending');

          return (
            <span key={i} style={{ display: 'contents' }}>
              <span className={`timeline-dot ${dotClass}`}
                    style={status === 'fail' ? { background: 'var(--color-error)' } :
                           status === 'warning' ? { background: 'var(--color-warning)' } :
                           status === 'skipped' ? { background: 'var(--color-text-muted)', opacity: 0.4 } : {}} />
              {i < steps.length - 1 && (
                <span className={`timeline-line ${dotClass === 'completed' ? 'completed' : ''}`}
                      style={status === 'fail' ? { background: 'var(--color-error)' } :
                             status === 'warning' ? { background: 'var(--color-warning)' } : {}} />
              )}
            </span>
          );
        })}
      </div>

      {/* Step Cards */}
      <div className="agent-trace">
        {steps.map((step) => {
          const isExpanded = expandedStep === step.agent_id;
          const status = step.status || 'pending';
          const meta = getStepMeta(step);

          return (
            <div
              key={step.agent_id}
              className={`agent-step stage-${meta.className} ${isExpanded ? 'expanded' : ''}`}
              onDoubleClick={() => handleStepDoubleClick(step)}
              style={{ animationDelay: `${step.agent_id * 0.08}s` }}
              title={step.summary ? 'Double-click to expand or collapse' : undefined}
            >
              <span className={`agent-step-number ${status}`}
                    style={status === 'skipped' ? { opacity: 0.45 } : {}}>
                {status === 'pass' ? '✓' :
                 status === 'warning' ? '!' :
                 status === 'fail' ? '✗' :
                 status === 'skipped' ? '–' :
                 loading ? '⋯' : step.agent_id}
              </span>

              <div className="agent-step-content">
                <div className="agent-step-name">
                  {step.agent}
                  {status !== 'pending' && (
                    <span className={`tag tag-${status === 'pass' ? 'green' : status === 'warning' ? 'amber' : status === 'fail' ? 'red' : 'gray'}`}
                          style={{ fontSize: '10px', padding: '0 6px', opacity: status === 'skipped' ? 0.6 : 1 }}>
                      {status}
                    </span>
                  )}
                </div>
                <div className="agent-step-subtitle">
                  <span className={`agent-step-stage stage-${meta.className}`}>{meta.stage}</span>
                  {meta.subtitle && <span>{meta.subtitle}</span>}
                </div>
                {step.summary && (
                  <div className="agent-step-summary">
                    {step.summary.length > 80 && !isExpanded
                      ? step.summary.slice(0, 80) + '...'
                      : step.summary}
                  </div>
                )}
                {step.agent === 'Difficulty Routing' && step.output && !loading && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
                    <span className={`tag ${step.output.difficulty === 'easy' ? 'tag-green' : step.output.difficulty === 'hard' ? 'tag-red' : 'tag-amber'}`}
                          style={{ fontSize: '10px', padding: '0 5px' }}>
                      {step.output.difficulty}
                    </span>
                    <span className={`tag ${step.output.verification_policy === 'strict' ? 'tag-amber' : 'tag-gray'}`}
                          style={{ fontSize: '10px', padding: '0 5px' }}>
                      {step.output.verification_policy} verify
                    </span>
                  </div>
                )}
                {isExpanded && step.output && (
                  <div className="agent-step-details">
                    <pre style={{
                      fontSize: 'var(--font-size-xs)',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      color: 'var(--color-text-muted)',
                      fontFamily: 'monospace',
                    }}>
                      {JSON.stringify(step.output, null, 2)}
                    </pre>
                  </div>
                )}
              </div>

              {step.duration_ms !== undefined && (
                <span className="agent-step-duration">{step.duration_ms}ms</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
