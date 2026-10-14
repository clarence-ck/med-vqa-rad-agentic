function fmt(ms) {
  if (ms == null) return null;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

export default function ComparisonToggle({ mode, onToggle, baselineMs, workflowMs, agenticMs }) {
  return (
    <div className="comparison-toggle-group">
      <div className="comparison-toggle">
        <button
          className={`comparison-toggle-btn ${mode === 'agentic' ? 'active' : ''}`}
          onClick={() => onToggle('agentic')}
        >
          🔗 Agentic Loop
          {agenticMs != null && (
            <span style={{ display: 'block', fontSize: '10px', opacity: 0.7, marginTop: '2px' }}>
              {fmt(agenticMs)}
            </span>
          )}
        </button>
        <button
          className={`comparison-toggle-btn ${mode === 'workflow' ? 'active' : ''}`}
          onClick={() => onToggle('workflow')}
        >
          🧭 Workflow Baseline
          {workflowMs != null && (
            <span style={{ display: 'block', fontSize: '10px', opacity: 0.7, marginTop: '2px' }}>
              {fmt(workflowMs)}
            </span>
          )}
        </button>
        <button
          className={`comparison-toggle-btn ${mode === 'baseline' ? 'active' : ''}`}
          onClick={() => onToggle('baseline')}
        >
          ⚡ Baseline (Single-shot)
          {baselineMs != null && (
            <span style={{ display: 'block', fontSize: '10px', opacity: 0.7, marginTop: '2px' }}>
              {fmt(baselineMs)}
            </span>
          )}
        </button>
      </div>
    </div>
  );
}
