export default function ExplanationCard({ result }) {
  if (!result) return null;

  const evidence = result.evidence_used || [];

  return (
    <div className="card animate-fade-in" style={{ animationDelay: '0.1s' }}>
      <div className="card-header">
        <span className="card-title">📋 Explanation</span>
      </div>
      <div className="card-content">
        <p className="explanation-text">{result.explanation}</p>
      </div>

      {evidence.length > 0 && (
        <div style={{ marginTop: 'var(--space-4)' }}>
          <div className="card-title" style={{ marginBottom: 'var(--space-2)' }}>Evidence Used</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
            {evidence.map((e, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-2)' }}>
                <span style={{ color: 'var(--color-accent)', flexShrink: 0, fontSize: '12px', marginTop: '2px' }}>▸</span>
                <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>{e}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {result.reasoning_trace && (
        <div style={{ marginTop: 'var(--space-4)' }}>
          <div className="card-title" style={{ marginBottom: 'var(--space-2)' }}>Reasoning Trace</div>
          <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-muted)', lineHeight: 1.7 }}>
            {result.reasoning_trace}
          </p>
        </div>
      )}
    </div>
  );
}
