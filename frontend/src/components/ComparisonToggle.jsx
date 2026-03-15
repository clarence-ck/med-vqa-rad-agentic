export default function ComparisonToggle({ mode, onToggle }) {
  return (
    <div className="comparison-toggle-group">
      <div className="comparison-toggle">
        <button
          className={`comparison-toggle-btn ${mode === 'agentic' ? 'active' : ''}`}
          onClick={() => onToggle('agentic')}
        >
          🔗 Agentic Pipeline
        </button>
        <button
          className={`comparison-toggle-btn ${mode === 'workflow' ? 'active' : ''}`}
          onClick={() => onToggle('workflow')}
        >
          🧭 Workflow Baseline
        </button>
        <button
          className={`comparison-toggle-btn ${mode === 'baseline' ? 'active' : ''}`}
          onClick={() => onToggle('baseline')}
        >
          ⚡ Baseline (Single-shot)
        </button>
      </div>
    </div>
  );
}
