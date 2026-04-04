 import { useEffect, useRef, useState } from 'react';
 
function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function fmtMs(ms) {
  if (ms == null) return '—';
  if (ms >= 60000) return `${(ms / 60000).toFixed(1)}m`;
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

function formatLabel(value) {
  return String(value || 'unknown').replace(/_/g, ' ');
}

function SystemPerformanceCard({ title, icon, accuracy, avgDuration, totalDuration, toneClass }) {
  return (
    <div className="card evaluation-system-card">
      <div className="card-header">
        <span className="card-title">{icon} {title}</span>
      </div>
      <div className="evaluation-system-stats">
        <div>
          <div className="evaluation-list-subtitle">Exact-match against Ground Truth</div>
          <div className="evaluation-metric-value">{formatPercent(accuracy)}</div>
        </div>
        <div>
          <div className="evaluation-list-subtitle">Avg runtime</div>
          <div className="evaluation-system-runtime">{fmtMs(avgDuration)}</div>
        </div>
        <div>
          <div className="evaluation-list-subtitle">Total runtime</div>
          <div className="evaluation-system-runtime" style={{ color: 'var(--color-text-muted)' }}>{fmtMs(totalDuration)}</div>
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

function RuntimeMetricStack({ baseline, workflow, agentic }) {
  return (
    <div className="evaluation-metric-stack">
      <span className="tag tag-amber">⚡ Baseline {fmtMs(baseline)}</span>
      <span className="tag tag-blue">🧭 Workflow {fmtMs(workflow)}</span>
      <span className="tag tag-green">🔗 Agentic {fmtMs(agentic)}</span>
    </div>
  );
}

function AccuracyBar({ label, value, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px' }}>
      <div style={{ width: '65px', color: 'var(--color-text-secondary)', textAlign: 'right' }}>{label}</div>
      <div style={{ flex: 1, backgroundColor: 'var(--color-border)', height: '10px', borderRadius: '5px', overflow: 'hidden' }}>
        <div style={{ width: `${value * 100}%`, height: '100%', backgroundColor: color, borderRadius: '5px' }}></div>
      </div>
      <div style={{ width: '35px', fontWeight: 600, textAlign: 'left', color: 'var(--color-text-primary)' }}>{Math.round(value * 100)}%</div>
    </div>
  );
}

export default function EvaluationView({ loading, evaluationData, onRefresh }) {
  const summary = evaluationData?.summary;
  const rows = evaluationData?.rows || [];
  const [selectedSampleIds, setSelectedSampleIds] = useState([]);
  const [isSampleDropdownOpen, setIsSampleDropdownOpen] = useState(false);
  const sampleDropdownRef = useRef(null);

  useEffect(() => {
    const handleDocumentClick = (event) => {
      if (sampleDropdownRef.current && !sampleDropdownRef.current.contains(event.target)) {
        setIsSampleDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleDocumentClick);
    return () => document.removeEventListener('mousedown', handleDocumentClick);
  }, []);

  const avgBaselineF1 = rows.length ? rows.reduce((acc, r) => acc + (r.baseline_token_f1 || 0), 0) / rows.length : 0;
  const avgWorkflowF1 = rows.length ? rows.reduce((acc, r) => acc + (r.workflow_token_f1 || 0), 0) / rows.length : 0;
  const avgAgenticF1 = rows.length ? rows.reduce((acc, r) => acc + (r.agentic_token_f1 || 0), 0) / rows.length : 0;
  const availableSampleIds = [...new Set(rows.map((row) => String(row.sample_id ?? '')).filter(Boolean))]
    .sort((left, right) => Number(left) - Number(right));
  const filteredRows = selectedSampleIds.length
    ? rows.filter((row) => selectedSampleIds.includes(String(row.sample_id ?? '')))
    : rows;

  const toggleSampleId = (sampleId) => {
    setSelectedSampleIds((current) => {
      if (current.includes(sampleId)) {
        return current.filter((id) => id !== sampleId);
      }

      if (current.length >= 5) {
        return current;
      }

      return [...current, sampleId];
    });
  };

  const clearSampleIdFilter = () => {
    setSelectedSampleIds([]);
  };

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
              totalDuration={summary.total_baseline_duration_ms}
              toneClass="tag-amber"
            />
            <SystemPerformanceCard
              title="Workflow"
              icon="🧭"
              accuracy={summary.workflow_accuracy}
              avgDuration={summary.avg_workflow_duration_ms}
              totalDuration={summary.total_workflow_duration_ms}
              toneClass="tag-blue"
            />
            <SystemPerformanceCard
              title="Agentic"
              icon="🔗"
              accuracy={summary.agentic_accuracy}
              avgDuration={summary.avg_agentic_duration_ms}
              totalDuration={summary.total_agentic_duration_ms}
              toneClass="tag-green"
            />
          </section>

          <section className="evaluation-metric-grid animate-fade-in">
            <div className="card">
              <div className="card-header"><span className="card-title">Total Cases</span></div>
              <div className="evaluation-metric-value">{summary.total_cases}</div>
            </div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">3-System Disagreement</span>
                <span className="tag tag-gray" data-tooltip="Percentage of cases where at least one of the three systems yielded a different result than the others." data-tooltip-align="left">info</span>
              </div>
              <div className="evaluation-metric-value">{formatPercent(summary.disagreement_rate)}</div>
            </div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">Avg Token F1 Score</span>
                <span className="tag tag-gray" data-tooltip="A partial-credit score. Measures precision and recall of exact word overlap between the generated answer and the Ground Truth." data-tooltip-align="left">info</span>
              </div>
              <AccuracyMetricStack
                baseline={avgBaselineF1}
                workflow={avgWorkflowF1}
                agentic={avgAgenticF1}
              />
            </div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">Wins</span>
                <span className="tag tag-gray" data-tooltip="A 'Win' happens when one system uniquely finds the correct answer. 'Ties' occur when multiple systems are correct, or when all systems fail." data-tooltip-align="left">info</span>
              </div>
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
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className="tag tag-gray">{summary.close_ended_cases} cases</span>
                  <span className="tag tag-gray" data-tooltip="Accuracy on questions with strict, fixed answers, such as yes/no or multiple choice." data-tooltip-align="left">info</span>
                </div>
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
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className="tag tag-gray">{summary.open_ended_cases} cases</span>
                  <span className="tag tag-gray" data-tooltip="A strict binary pass/fail grade. Evaluates if the core clinical concept was successfully identified, regardless of filler phrasing." data-tooltip-align="left">info</span>
                </div>
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
                <span className="card-title">Accuracy by Question Type</span>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className="tag tag-gray">exact match</span>
                  <span className="tag tag-gray" data-tooltip="A breakdown of how often each pipeline's answer perfectly matched the Ground Truth across different clinical question categories." data-tooltip-align="left">info</span>
                </div>
              </div>
              <div className="evaluation-table-wrap" style={{ margin: '0 -24px -24px -24px' }}>
                <table className="evaluation-table" style={{ margin: 0, borderTop: '1px solid var(--color-border)' }}>
                  <thead>
                    <tr>
                      <th>Question Type</th>
                      <th>Total Cases</th>
                      <th>Baseline</th>
                      <th>Workflow</th>
                      <th>Agentic</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.by_question_type.map((item) => (
                      <tr key={item.question_type}>
                        <td style={{ textTransform: 'capitalize', fontWeight: 500, color: 'var(--color-text-primary)' }}>
                          {item.question_type.replace(/_/g, ' ')}
                        </td>
                        <td><span className="tag tag-gray">{item.count}</span></td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <div style={{ width: '40px', backgroundColor: 'var(--color-border)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                              <div style={{ width: `${item.baseline_accuracy * 100}%`, height: '100%', backgroundColor: 'var(--color-warning)', borderRadius: '3px' }}></div>
                            </div>
                            <span style={{ fontSize: '11px', fontWeight: 600 }}>{formatPercent(item.baseline_accuracy)}</span>
                          </div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <div style={{ width: '40px', backgroundColor: 'var(--color-border)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                              <div style={{ width: `${item.workflow_accuracy * 100}%`, height: '100%', backgroundColor: 'var(--color-accent)', borderRadius: '3px' }}></div>
                            </div>
                            <span style={{ fontSize: '11px', fontWeight: 600 }}>{formatPercent(item.workflow_accuracy)}</span>
                          </div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <div style={{ width: '40px', backgroundColor: 'var(--color-border)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                              <div style={{ width: `${item.agentic_accuracy * 100}%`, height: '100%', backgroundColor: 'var(--color-success)', borderRadius: '3px' }}></div>
                            </div>
                            <span style={{ fontSize: '11px', fontWeight: 600 }}>{formatPercent(item.agentic_accuracy)}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <span className="card-title">Recent Evaluated Cases</span>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className="tag tag-gray">
                    {filteredRows.length} of {rows.length} shown
                  </span>
                  <span className="tag tag-gray">
                    {selectedSampleIds.length} selected
                  </span>
                </div>
              </div>
              <div ref={sampleDropdownRef} style={{ padding: '0 24px 16px 24px', position: 'relative' }}>
                <button
                  type="button"
                  className="form-select"
                  onClick={() => setIsSampleDropdownOpen((open) => !open)}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}
                >
                  <span>
                    {selectedSampleIds.length === 0
                      ? 'Filter by sample IDs'
                      : `${selectedSampleIds.length} selected`}
                  </span>
                  <span aria-hidden="true">▾</span>
                </button>

                {isSampleDropdownOpen && (
                  <div
                    style={{
                      marginTop: '8px',
                      border: '1px solid var(--color-border)',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--color-bg-secondary)',
                      boxShadow: '0 16px 36px rgba(0, 0, 0, 0.28)',
                      padding: '12px',
                      maxHeight: '320px',
                      overflowY: 'auto',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', alignItems: 'center', marginBottom: '12px' }}>
                      <div className="evaluation-list-subtitle">
                        Select up to 5 sample IDs.
                      </div>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={clearSampleIdFilter}
                        disabled={!selectedSampleIds.length}
                        style={{ padding: '6px 10px', fontSize: '12px' }}
                      >
                        Clear
                      </button>
                    </div>

                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
                      {selectedSampleIds.length ? (
                        selectedSampleIds.map((sampleId) => (
                          <span key={sampleId} className="tag tag-green">
                            Sample ID {sampleId}
                          </span>
                        ))
                      ) : (
                        <span className="tag tag-gray">All sample IDs</span>
                      )}
                    </div>

                    <div style={{ display: 'grid', gap: '8px' }}>
                      {availableSampleIds.map((sampleId) => {
                        const checked = selectedSampleIds.includes(sampleId);
                        const disabled = !checked && selectedSampleIds.length >= 5;

                        return (
                          <label
                            key={sampleId}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '10px',
                              padding: '8px 10px',
                              borderRadius: 'var(--radius-sm)',
                              border: '1px solid var(--color-border)',
                              background: disabled && !checked ? 'var(--color-bg-input)' : 'var(--color-bg-card)',
                              opacity: disabled && !checked ? 0.6 : 1,
                              cursor: disabled && !checked ? 'not-allowed' : 'pointer',
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              disabled={disabled}
                              onChange={() => toggleSampleId(sampleId)}
                            />
                            <span>Sample ID {sampleId}</span>
                          </label>
                        );
                      })}
                    </div>

                    {selectedSampleIds.length >= 5 && (
                      <div className="evaluation-list-subtitle" style={{ marginTop: '10px', color: 'var(--color-warning)' }}>
                        Maximum of 5 sample IDs selected.
                      </div>
                    )}
                  </div>
                )}
              </div>
              <div className="evaluation-table-wrap" style={{ maxHeight: '720px', overflowY: 'auto' }}>
                <table className="evaluation-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Question & Truth</th>
                      <th>Answers</th>
                      <th>Winner</th>
                      <th>Agentic Retriever</th>
                      <th>Agentic Verify</th>
                      <th>Agentic Signals</th>
                      <th>Runtime</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRows.map((row) => (
                      <tr key={`${row.sample_id}-${row.cached_at || 'na'}`}>
                        <td>{row.sample_id}</td>
                        <td>
                          <div className="evaluation-question-cell" style={{ maxWidth: '280px' }}>
                            <div className="evaluation-list-title">{row.question || '—'}</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', margin: '6px 0 4px 0' }}>
                              <span className="tag tag-gray" style={{ fontSize: '10px' }}>
                                Original Question Type: {formatLabel(row.original_question_type)}
                              </span>
                              <span className={`tag ${row.agentic_question_type ? 'tag-blue' : 'tag-gray'}`} style={{ fontSize: '10px' }}>
                                Agentic Reclass Type: {formatLabel(row.agentic_question_type || row.original_question_type)}
                              </span>
                              <span className="tag tag-gray" style={{ fontSize: '10px' }}>
                                Ground Truth Answer Style: {formatLabel(row.answer_style)}
                              </span>
                            </div>
                            <div className="evaluation-list-title" style={{ fontSize: '11px', color: 'var(--color-success)' }}>
                              GT: {row.ground_truth_answer}
                            </div>
                          </div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', maxWidth: '300px' }}>
                            <div>
                              <span className="tag tag-amber" style={{ fontSize: '9px', padding: '1px 5px', minWidth: '55px', justifyContent: 'center', marginRight: '6px' }}>Baseline</span>
                              <span style={{ color: row.baseline_correct ? 'var(--color-success)' : 'inherit', textDecoration: !row.baseline_correct && row.baseline_answer ? 'line-through' : 'none' }}>{row.baseline_answer || '—'}</span>
                            </div>
                            <div>
                              <span className="tag tag-blue" style={{ fontSize: '9px', padding: '1px 5px', minWidth: '55px', justifyContent: 'center', marginRight: '6px' }}>Workflow</span>
                              <span style={{ color: row.workflow_correct ? 'var(--color-success)' : 'inherit', textDecoration: !row.workflow_correct && row.workflow_answer ? 'line-through' : 'none' }}>{row.workflow_answer || '—'}</span>
                            </div>
                            <div>
                              <span className="tag tag-green" style={{ fontSize: '9px', padding: '1px 5px', minWidth: '55px', justifyContent: 'center', marginRight: '6px' }}>Agentic</span>
                              <span style={{ color: row.agentic_correct ? 'var(--color-success)' : 'inherit', textDecoration: !row.agentic_correct && row.agentic_answer ? 'line-through' : 'none' }}>{row.agentic_answer || '—'}</span>
                            </div>
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
                            <span className={`tag ${getLevelTagClass(row.confidence)}`}>
                              confidence {(row.confidence || 'low').toLowerCase()}
                            </span>
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
                    {filteredRows.length === 0 && (
                      <tr>
                        <td colSpan="8" style={{ textAlign: 'center', color: 'var(--color-text-muted)' }}>
                          No evaluated cases match the selected sample IDs.
                        </td>
                      </tr>
                    )}
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
