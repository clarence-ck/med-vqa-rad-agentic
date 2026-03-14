export default function ComparisonToggle({ mode, onToggle }) {
  const descriptions = {
    agentic: 'Multi-stage pipeline with image interpretation, question analysis, knowledge support, reasoning, and verification.',
    workflow: 'Hand-designed non-agentic workflow that uses structured intermediate steps but no verifier-driven agent loop.',
    baseline: 'Single multimodal prompt that answers directly from the image and question without decomposition or verification.',
  };

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
      <div className="comparison-toggle-help">
        {descriptions[mode]}
      </div>
    </div>
  );
}
