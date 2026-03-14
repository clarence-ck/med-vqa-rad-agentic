function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function getWinnerTagClass(winner) {
  if (winner === 'agentic') return 'tag-green';
  if (winner === 'workflow') return 'tag-blue';
  if (winner === 'baseline') return 'tag-amber';
  return 'tag-gray';
}

export default function EvaluationView({ loading, evaluationData, activeSplit, onSplitChange, onRefresh }) {
  const summary = evaluationData?.summary;
  const rows = evaluationData?.rows || [];

  return (
    <main className="evaluation-layout">
      <section className="evaluation-header-strip">
        <div>
          <h1 className="evaluation-title">Evaluation Dashboard</h1>
          <p className="evaluation-subtitle">Aggregate exact-match metrics from cached Case Explorer runs.</p>
        </div>
        <div className="evaluation-actions">
          <select
            className="form-select evaluation-split-select"
            value={activeSplit || 'all'}
            onChange={(e) => onSplitChange?.(e.target.value)}
          >
            <option value="test">Test — recommended final reporting split</option>
            <option value="train">Train — development-only evaluation</option>
            <option value="all">All — mixed / not for reporting</option>
          </select>
          <button className="btn btn-secondary" onClick={onRefresh} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh Metrics'}
          </button>
        </div>
      </section>

      <section className="split-callout-card">
        {activeSplit === 'test' && (
          <div className="split-warning-banner split-warning-green">
            <span>✅</span>
            <span>You are evaluating on the <strong>test</strong> split, which is the recommended mode for final reporting.</span>
          </div>
        )}
        {activeSplit === 'train' && (
          <div className="split-warning-banner split-warning-amber">
            <span>⚠️</span>
            <span>You are viewing <strong>development-only</strong> metrics. Use the <strong>test</strong> split for final benchmark numbers.</span>
          </div>
        )}
        {activeSplit === 'all' && (
          <div className="split-warning-banner">
            <span>ℹ️</span>
            <span>You are mixing <strong>train</strong> and <strong>test</strong> results. This is useful for monitoring, but not for final reporting.</span>
          </div>
        )}
      </section>

      {!summary && !loading && (
        <div className="empty-state" style={{ flex: 1 }}>
          <span className="icon">📊</span>
          <div className="empty-state-title">No Evaluation Data Yet</div>
          <div className="empty-state-text">Run a few cases in Case Explorer first, then come back here for aggregated results.</div>
        </div>
      )}

      {loading && (
        <div className="loading-container" style={{ flex: 1 }}>
          <div className="loading-spinner"></div>
          <div className="loading-text">Loading evaluation metrics...</div>
        </div>
      )}

      {summary && !loading && (
        <>
          <section className="evaluation-metric-grid animate-fade-in">
            <div className="card">
              <div className="card-header"><span className="card-title">Total Cases</span></div>
              <div className="evaluation-metric-value">{summary.total_cases}</div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">Baseline Exact-Match</span></div>
              <div className="evaluation-metric-value">{formatPercent(summary.baseline_accuracy)}</div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">Workflow Exact-Match</span></div>
              <div className="evaluation-metric-value">{formatPercent(summary.workflow_accuracy)}</div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">Agentic Exact-Match</span></div>
              <div className="evaluation-metric-value">{formatPercent(summary.agentic_accuracy)}</div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">3-System Disagreement</span></div>
              <div className="evaluation-metric-value">{formatPercent(summary.disagreement_rate)}</div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">Wins</span></div>
              <div className="evaluation-metric-stack">
                <span className="tag tag-amber">Baseline {summary.baseline_wins}</span>
                <span className="tag tag-blue">Workflow {summary.workflow_wins}</span>
                <span className="tag tag-green">Agentic {summary.agentic_wins}</span>
                <span className="tag tag-gray">Ties {summary.ties}</span>
              </div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">Avg Duration</span></div>
              <div className="evaluation-metric-value">{summary.avg_duration_ms}ms</div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">Split Counts</span></div>
              <div className="evaluation-metric-stack">
                {Object.entries(summary.split_counts || {}).map(([split, count]) => (
                  <span key={split} className="tag tag-gray">{split} {count}</span>
                ))}
              </div>
            </div>
          </section>

          <section className="evaluation-two-col animate-fade-in">
            <div className="card">
              <div className="card-header">
                <span className="card-title">Exact-Match by Question Type</span>
                <span className="tag tag-gray">exact match</span>
              </div>
              <div className="evaluation-list">
                {summary.by_question_type.map((item) => (
                  <div key={item.question_type} className="evaluation-list-row">
                    <div>
                      <div className="evaluation-list-title">{item.question_type.replace(/_/g, ' ')}</div>
                      <div className="evaluation-list-subtitle">{item.count} cases</div>
                    </div>
                    <div className="evaluation-list-metrics">
                      <span className="tag tag-amber">B {formatPercent(item.baseline_accuracy)}</span>
                      <span className="tag tag-blue">W {formatPercent(item.workflow_accuracy)}</span>
                      <span className="tag tag-green">A {formatPercent(item.agentic_accuracy)}</span>
                      <span className="tag tag-gray">D {formatPercent(item.disagreement_rate)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <span className="card-title">Recent Evaluated Cases</span>
              </div>
              <div className="evaluation-table-wrap">
                <table className="evaluation-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Type</th>
                      <th>Winner</th>
                      <th>Split</th>
                      <th>Verify</th>
                      <th>Consistency</th>
                      <th>Cross-modal</th>
                      <th>Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(0, 12).map((row) => (
                      <tr key={`${row.sample_id}-${row.cached_at || 'na'}`}>
                        <td>{row.sample_id}</td>
                        <td>{row.question_type?.replace(/_/g, ' ')}</td>
                        <td><span className={`tag ${getWinnerTagClass(row.winner)}`}>{row.winner.replace(/_/g, ' ')}</span></td>
                        <td>{row.split}</td>
                        <td>{row.verification_status?.replace(/_/g, ' ')}</td>
                        <td>{row.self_consistency_level}</td>
                        <td>{row.cross_modal_agreement_level}</td>
                        <td>{row.duration_ms}ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </>
      )}
    </main>
  );
}
