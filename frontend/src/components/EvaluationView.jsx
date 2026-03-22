function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function fmtMs(ms) {
  if (ms == null) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function getWinnerTagClass(winner) {
  if (winner === 'agentic') return 'tag-green';
  if (winner === 'workflow') return 'tag-blue';
  if (winner === 'baseline') return 'tag-amber';
  return 'tag-gray';
}

function getLevelTagClass(level) {
  const normalized = String(level || 'unknown').toLowerCase();
  if (normalized === 'high' || normalized === 'verified') return 'tag-green';
  if (normalized === 'medium' || normalized === 'partially_verified') return 'tag-amber';
  if (normalized === 'low' || normalized === 'unsupported' || normalized === 'low_confidence') return 'tag-red';
  return 'tag-gray';
}

function SystemPerformanceCard({ title, icon, accuracy, avgDuration, toneClass }) {
  return (
    <div className="card evaluation-system-card">
      <div className="card-header">
        <span className="card-title">{icon} {title}</span>
        <span className={`tag ${toneClass}`}>{formatPercent(accuracy)}</span>
      </div>
      <div className="evaluation-system-stats">
        <div>
          <div className="evaluation-list-subtitle">Exact-match</div>
          <div className="evaluation-metric-value">{formatPercent(accuracy)}</div>
        </div>
        <div>
          <div className="evaluation-list-subtitle">Avg runtime</div>
          <div className="evaluation-system-runtime">{fmtMs(avgDuration)}</div>
        </div>
      </div>
    </div>
  );
}

function AccuracyMetricStack({ baseline, workflow, agentic }) {
  return (
    <div className="evaluation-metric-stack">
      <span className="tag tag-amber">Baseline {formatPercent(baseline)}</span>
      <span className="tag tag-blue">Workflow {formatPercent(workflow)}</span>
      <span className="tag tag-green">Agentic {formatPercent(agentic)}</span>
    </div>
  );
}

export default function EvaluationView({ loading, evaluationData, onRefresh }) {
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
          <button className="btn btn-secondary" onClick={onRefresh} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh Metrics'}
          </button>
        </div>
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
          <section className="evaluation-system-grid animate-fade-in">
            <SystemPerformanceCard
              title="Baseline"
              icon="⚡"
              accuracy={summary.baseline_accuracy}
              avgDuration={summary.avg_baseline_duration_ms}
              toneClass="tag-amber"
            />
            <SystemPerformanceCard
              title="Workflow"
              icon="🧭"
              accuracy={summary.workflow_accuracy}
              avgDuration={summary.avg_workflow_duration_ms}
              toneClass="tag-blue"
            />
            <SystemPerformanceCard
              title="Agentic"
              icon="🔗"
              accuracy={summary.agentic_accuracy}
              avgDuration={summary.avg_agentic_duration_ms}
              toneClass="tag-green"
            />
          </section>

          <section className="evaluation-metric-grid animate-fade-in">
            <div className="card">
              <div className="card-header"><span className="card-title">Total Cases</span></div>
              <div className="evaluation-metric-value">{summary.total_cases}</div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">3-System Disagreement</span></div>
              <div className="evaluation-metric-value">{formatPercent(summary.disagreement_rate)}</div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">Avg End-to-End Runtime</span></div>
              <div className="evaluation-metric-value">{fmtMs(summary.avg_duration_ms)}</div>
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
              <div className="card-header">
                <span className="card-title">Close-ended Accuracy</span>
                <span className="tag tag-gray">{summary.close_ended_cases} cases</span>
              </div>
              <AccuracyMetricStack
                baseline={summary.baseline_close_ended_accuracy}
                workflow={summary.workflow_close_ended_accuracy}
                agentic={summary.agentic_close_ended_accuracy}
              />
            </div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">Open-ended Accuracy</span>
                <span className="tag tag-gray">{summary.open_ended_cases} cases</span>
              </div>
              <AccuracyMetricStack
                baseline={summary.baseline_open_ended_accuracy}
                workflow={summary.workflow_open_ended_accuracy}
                agentic={summary.agentic_open_ended_accuracy}
              />
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
                      <span className="tag tag-amber">Baseline {formatPercent(item.baseline_accuracy)}</span>
                      <span className="tag tag-blue">Workflow {formatPercent(item.workflow_accuracy)}</span>
                      <span className="tag tag-green">Agentic {formatPercent(item.agentic_accuracy)}</span>
                      <span className="tag tag-gray">Disagree {formatPercent(item.disagreement_rate)}</span>
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
                      <th>Question</th>
                      <th>Winner</th>
                      <th>Retriever</th>
                      <th>Verify</th>
                      <th>Signals</th>
                      <th>Runtime</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(0, 12).map((row) => (
                      <tr key={`${row.sample_id}-${row.cached_at || 'na'}`}>
                        <td>{row.sample_id}</td>
                        <td>
                          <div className="evaluation-question-cell">
                            <div className="evaluation-list-title">{row.question || '—'}</div>
                            <div className="evaluation-list-subtitle">{row.question_type?.replace(/_/g, ' ') || 'unknown'}</div>
                          </div>
                        </td>
                        <td><span className={`tag ${getWinnerTagClass(row.winner)}`}>{row.winner.replace(/_/g, ' ')}</span></td>
                        <td>
                          <div className="evaluation-signal-stack">
                            <span className={`tag ${row.knowledge_retriever_triggered ? 'tag-green' : 'tag-gray'}`}>
                              {row.knowledge_retriever_triggered
                                ? (row.knowledge_retriever_status || 'triggered')
                                : 'skipped'}
                            </span>
                            <span className="evaluation-list-subtitle">facts {row.knowledge_key_facts_count ?? 0}</span>
                          </div>
                        </td>
                        <td>
                          <span className={`tag ${getLevelTagClass(row.verification_status)}`}>
                            {row.verification_status?.replace(/_/g, ' ') || 'unknown'}
                          </span>
                        </td>
                        <td>
                          <div className="evaluation-signal-stack">
                            <span className={`tag ${getLevelTagClass(row.self_consistency_level)}`}>
                              consistency {row.self_consistency_level}
                            </span>
                            <span className={`tag ${getLevelTagClass(row.cross_modal_agreement_level)}`}>
                              cross-modal {row.cross_modal_agreement_level}
                            </span>
                          </div>
                        </td>
                        <td>
                          <div className="evaluation-runtime-stack">
                            <div>⚡ {fmtMs(row.baseline_duration_ms)}</div>
                            <div>🧭 {fmtMs(row.workflow_duration_ms)}</div>
                            <div>🔗 {fmtMs(row.agentic_duration_ms)}</div>
                          </div>
                        </td>
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
