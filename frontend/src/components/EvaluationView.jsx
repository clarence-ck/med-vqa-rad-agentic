 import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
 
function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function fmtMs(ms) {
  if (ms == null) return '—';
  if (ms >= 60000) {
    const totalSeconds = Math.round(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}m ${seconds}s`;
  }
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

function getDifficultyTagClass(difficulty) {
  const normalized = String(difficulty || 'unknown').toLowerCase();
  if (normalized === 'easy') return 'tag-green';
  if (normalized === 'medium') return 'tag-amber';
  if (normalized === 'hard') return 'tag-red';
  return 'tag-gray';
}

function getVerificationPolicyTagClass(policy) {
  return String(policy || 'standard').toLowerCase() === 'strict' ? 'tag-amber' : 'tag-gray';
}

function getRetrievalTagClass(needsRetrieval) {
  return needsRetrieval ? 'tag-green' : 'tag-gray';
}

function getRoutingProcedureTooltip(difficulty) {
  const normalized = String(difficulty || 'unknown').toLowerCase();

  if (normalized === 'easy') {
    return 'Procedure: interpret image → analyze question → skip retrieval → use standard verification → finalize.';
  }

  if (normalized === 'medium') {
    return 'Procedure: interpret image → analyze question → retrieve knowledge → use standard verification → add disclaimer if needed.';
  }

  if (normalized === 'hard') {
    return 'Procedure: interpret image → analyze question → retrieve knowledge → use strict verification → flag for review if needed.';
  }

  return 'Procedure: interpret image → analyze question → route difficulty → retrieve knowledge only when needed → verify the final answer.';
}

function getBreakdownTooltip(mode) {
  if (mode === 'difficulty_type') {
    return 'Breakdown by Difficulty Router output. Parent rows show overall difficulty accuracy, and expanding a difficulty row reveals question-type accuracy within that bucket. easy skips retrieval and uses standard verification; medium retrieves knowledge, uses standard verification, and can add a disclaimer; hard retrieves knowledge, uses strict verification, and can flag the case for review.';
  }

  return 'Breakdown by question type. Close-ended questions use exact yes/no only. Open-ended questions use PubMedBERT semantic similarity ≥ 0.75 to capture meaning equivalence.';
}

const QUESTION_TYPE_ORDER = [
  'anatomy_identification',
  'modality_identification',
  'yes_no',
  'counting',
  'location',
  'abnormality_description',
  'other',
];

const DIFFICULTY_ORDER = ['easy', 'medium', 'hard', 'unknown'];

function normalizeBreakdownLabel(value) {
  return String(value || 'unknown').trim().toLowerCase() || 'unknown';
}

function createBreakdownBucket(label) {
  return {
    label,
    count: 0,
    baseline_correct: 0,
    workflow_correct: 0,
    agentic_correct: 0,
    disagreement_count: 0,
  };
}

function accumulateBreakdownBucket(bucket, row) {
  bucket.count += 1;
  bucket.baseline_correct += row.baseline_correct ? 1 : 0;
  bucket.workflow_correct += row.workflow_correct ? 1 : 0;
  bucket.agentic_correct += row.agentic_correct ? 1 : 0;
  bucket.disagreement_count += row.disagreement ? 1 : 0;
}

function finalizeBreakdownBucket(bucket) {
  return {
    label: bucket.label,
    count: bucket.count,
    baseline_accuracy: bucket.count ? bucket.baseline_correct / bucket.count : 0,
    workflow_accuracy: bucket.count ? bucket.workflow_correct / bucket.count : 0,
    agentic_accuracy: bucket.count ? bucket.agentic_correct / bucket.count : 0,
    disagreement_rate: bucket.count ? bucket.disagreement_count / bucket.count : 0,
  };
}

function compareLabels(left, right, preferredOrder = []) {
  const leftIndex = preferredOrder.indexOf(left);
  const rightIndex = preferredOrder.indexOf(right);

  if (leftIndex !== -1 || rightIndex !== -1) {
    if (leftIndex === -1) return 1;
    if (rightIndex === -1) return -1;
    if (leftIndex !== rightIndex) return leftIndex - rightIndex;
  }

  return left.localeCompare(right);
}

function buildDifficultyBreakdown(rows) {
  const difficultyBuckets = new Map();
  const questionTypeBucketsByDifficulty = new Map();

  rows.forEach((row) => {
    const difficulty = normalizeBreakdownLabel(row.routing_difficulty);
    const questionType = normalizeBreakdownLabel(row.question_type || row.original_question_type);

    if (!difficultyBuckets.has(difficulty)) {
      difficultyBuckets.set(difficulty, createBreakdownBucket(difficulty));
      questionTypeBucketsByDifficulty.set(difficulty, new Map());
    }

    accumulateBreakdownBucket(difficultyBuckets.get(difficulty), row);

    const questionTypeBuckets = questionTypeBucketsByDifficulty.get(difficulty);
    if (!questionTypeBuckets.has(questionType)) {
      questionTypeBuckets.set(questionType, createBreakdownBucket(questionType));
    }

    accumulateBreakdownBucket(questionTypeBuckets.get(questionType), row);
  });

  return Array.from(difficultyBuckets.values())
    .sort((left, right) => compareLabels(left.label, right.label, DIFFICULTY_ORDER))
    .map((bucket) => {
      const questionTypeBuckets = Array.from(questionTypeBucketsByDifficulty.get(bucket.label)?.values() || [])
        .sort((left, right) => compareLabels(left.label, right.label, QUESTION_TYPE_ORDER))
        .map(finalizeBreakdownBucket);

      return {
        ...finalizeBreakdownBucket(bucket),
        question_type_breakdown: questionTypeBuckets,
      };
    });
}

function BreakdownAccuracyCell({ value, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div style={{ width: '40px', backgroundColor: 'var(--color-border)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
        <div style={{ width: `${value * 100}%`, height: '100%', backgroundColor: color, borderRadius: '3px' }}></div>
      </div>
      <span style={{ fontSize: '11px', fontWeight: 600 }}>{formatPercent(value)}</span>
    </div>
  );
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
          <div className="evaluation-list-subtitle">Overall Accuracy (Exact yes/no + PubMedBERT)</div>
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
  const [breakdownMode, setBreakdownMode] = useState('question_type');
  const [collapsedDifficultyGroups, setCollapsedDifficultyGroups] = useState([]);
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
  const isDifficultyMode = breakdownMode === 'difficulty_type';
  const questionBreakdownRows = summary?.by_question_type || [];
  const difficultyBreakdownRows = useMemo(() => buildDifficultyBreakdown(rows), [rows]);
  const breakdownRows = isDifficultyMode ? difficultyBreakdownRows : questionBreakdownRows;
  const breakdownTitle = isDifficultyMode ? 'Accuracy by Difficulty Type' : 'Accuracy by Question Type';
  const breakdownPrimaryLabel = isDifficultyMode ? 'Difficulty / Question Type' : 'Question Type';
  const breakdownContextTag = isDifficultyMode ? 'Routing' : 'PubMedBERT';
  const breakdownTooltip = getBreakdownTooltip(breakdownMode);
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

  const toggleDifficultyGroup = (groupLabel) => {
    setCollapsedDifficultyGroups((current) => (
      current.includes(groupLabel)
        ? current.filter((label) => label !== groupLabel)
        : [...current, groupLabel]
    ));
  };

  return (
    <main className="evaluation-layout">
      <section className="evaluation-header-strip">
        <div>
          <h1 className="evaluation-title">Evaluation Dashboard</h1>
          <p className="evaluation-subtitle">Aggregate accuracy metrics from cached Case Explorer runs. Close-ended: exact yes/no only. Open-ended: PubMedBERT semantic similarity ≥ 0.75.</p>
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
                <span className="tag tag-gray" data-tooltip="Percentage of cases where at least one pair of system outputs falls below the PubMedBERT semantic-match threshold (0.75), so the three answers are not all semantically aligned." data-tooltip-align="left">info</span>
              </div>
              <div className="evaluation-metric-value">{formatPercent(summary.disagreement_rate)}</div>
            </div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">Lexical Overlap</span>
                <span className="tag tag-gray" data-tooltip="Average token F1 overlap score (0-1) across all cases. This is a surface-level diagnostic similarity metric only and is not used for correctness. Compare with PubMedBERT Accuracy, which uses embedding similarity with a 0.75 threshold." data-tooltip-align="left">info</span>
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
                <span className="tag tag-gray" data-tooltip="A 'Win' means exactly one of the three systems is correct. Ties happen when two or more systems are correct, or when all three are wrong." data-tooltip-align="left">info</span>
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
                  <span className="tag tag-gray" data-tooltip="Accuracy on cases whose ground-truth answer is exactly yes or no. No fallback is used." data-tooltip-align="left">info</span>
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
                <span className="card-title">PubMedBERT Accuracy (Open-ended)</span>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className="tag tag-gray">{summary.open_ended_cases} cases</span>
                  <span className="tag tag-gray" data-tooltip="PubMedBERT Accuracy: Percentage of open-ended answers whose embedding similarity to the ground truth is ≥ 0.75. Uses a PubMedBERT-based sentence-transformer tuned for biomedical semantic similarity. Not based on token F1." data-tooltip-align="left">info</span>
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
                <span className="card-title">{breakdownTitle}</span>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                  <select
                    className="form-select"
                    value={breakdownMode}
                    onChange={(event) => setBreakdownMode(event.target.value)}
                    aria-label="Select evaluation breakdown mode"
                    style={{ minWidth: '190px', cursor: 'pointer' }}
                  >
                    <option value="question_type">Question Type</option>
                    <option value="difficulty_type">Difficulty Type</option>
                  </select>
                  <span className={`tag ${isDifficultyMode ? 'tag-green' : 'tag-gray'}`}>{breakdownContextTag}</span>
                  <span className="tag tag-gray" data-tooltip={breakdownTooltip} data-tooltip-align="left">info</span>
                </div>
              </div>
              <div className="evaluation-table-wrap" style={{ margin: '0 -24px -24px -24px' }}>
                <table className="evaluation-table" style={{ margin: 0, borderTop: '1px solid var(--color-border)' }}>
                  <thead>
                    <tr>
                      <th>{breakdownPrimaryLabel}</th>
                      <th>Total Cases</th>
                      <th>Baseline</th>
                      <th>Workflow</th>
                      <th>Agentic</th>
                    </tr>
                  </thead>
                  <tbody>
                    {!isDifficultyMode && breakdownRows.map((item) => (
                      <tr key={item.question_type || item.label}>
                        <td style={{ fontWeight: 500, color: 'var(--color-text-primary)' }}>
                          {formatLabel(item.question_type || item.label)}
                        </td>
                        <td><span className="tag tag-gray">{item.count}</span></td>
                        <td><BreakdownAccuracyCell value={item.baseline_accuracy} color="var(--color-warning)" /></td>
                        <td><BreakdownAccuracyCell value={item.workflow_accuracy} color="var(--color-accent)" /></td>
                        <td><BreakdownAccuracyCell value={item.agentic_accuracy} color="var(--color-success)" /></td>
                      </tr>
                    ))}
                    {isDifficultyMode && breakdownRows.map((difficultyGroup) => {
                      const isCollapsed = collapsedDifficultyGroups.includes(difficultyGroup.label);
                      const typeCount = difficultyGroup.question_type_breakdown.length;
                      const typeCountLabel = typeCount === 1 ? '1 question type' : `${typeCount} question types`;

                      return (
                        <Fragment key={difficultyGroup.label}>
                          <tr style={{ backgroundColor: 'var(--color-bg-secondary)' }}>
                            <td style={{ fontWeight: 600, color: 'var(--color-text-primary)' }}>
                              <button
                                type="button"
                                onClick={() => toggleDifficultyGroup(difficultyGroup.label)}
                                aria-expanded={!isCollapsed}
                                title={isCollapsed ? 'Expand difficulty group' : 'Collapse difficulty group'}
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '8px',
                                  width: '100%',
                                  background: 'transparent',
                                  border: 'none',
                                  padding: 0,
                                  color: 'inherit',
                                  font: 'inherit',
                                  cursor: 'pointer',
                                  textAlign: 'left',
                                }}
                              >
                                <span aria-hidden="true" style={{ width: '16px', display: 'inline-flex', justifyContent: 'center' }}>
                                  {isCollapsed ? '▸' : '▾'}
                                </span>
                                <span className={`tag ${getDifficultyTagClass(difficultyGroup.label)}`}>
                                  {formatLabel(difficultyGroup.label)}
                                </span>
                                <span className="tag tag-gray">{typeCountLabel}</span>
                              </button>
                            </td>
                            <td><span className="tag tag-gray">{difficultyGroup.count}</span></td>
                            <td><BreakdownAccuracyCell value={difficultyGroup.baseline_accuracy} color="var(--color-warning)" /></td>
                            <td><BreakdownAccuracyCell value={difficultyGroup.workflow_accuracy} color="var(--color-accent)" /></td>
                            <td><BreakdownAccuracyCell value={difficultyGroup.agentic_accuracy} color="var(--color-success)" /></td>
                          </tr>
                          {!isCollapsed && difficultyGroup.question_type_breakdown.map((item) => (
                            <tr key={`${difficultyGroup.label}-${item.label}`}>
                              <td style={{ paddingLeft: '24px', color: 'var(--color-text-secondary)' }}>
                                ↳ {formatLabel(item.label)}
                              </td>
                              <td><span className="tag tag-gray">{item.count}</span></td>
                              <td><BreakdownAccuracyCell value={item.baseline_accuracy} color="var(--color-warning)" /></td>
                              <td><BreakdownAccuracyCell value={item.workflow_accuracy} color="var(--color-accent)" /></td>
                              <td><BreakdownAccuracyCell value={item.agentic_accuracy} color="var(--color-success)" /></td>
                            </tr>
                          ))}
                        </Fragment>
                      );
                    })}
                    {breakdownRows.length === 0 && (
                      <tr>
                        <td colSpan="5" style={{ textAlign: 'center', color: 'var(--color-text-muted)' }}>
                          No breakdown data available.
                        </td>
                      </tr>
                    )}
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
                      boxShadow: 'var(--shadow-lg)',
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
                      <th>Routing</th>
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
                            <span
                              className={`tag ${getDifficultyTagClass(row.routing_difficulty)}`}
                              data-tooltip={getRoutingProcedureTooltip(row.routing_difficulty)}
                              data-tooltip-align="left"
                            >
                              {formatLabel(row.routing_difficulty)}
                            </span>
                            <span className={`tag ${getRetrievalTagClass(row.routing_needs_retrieval)}`}>
                              retrieval {row.routing_needs_retrieval ? 'yes' : 'no'}
                            </span>
                            <span className={`tag ${getVerificationPolicyTagClass(row.routing_verification_policy)}`}>
                              {formatLabel(row.routing_verification_policy)} verify
                            </span>
                            <span
                              className="tag tag-gray"
                              data-tooltip={[
                                row.routing_fallback_policy ? `Fallback: ${formatLabel(row.routing_fallback_policy)}` : null,
                                row.routing_rationale ? `Routing: ${row.routing_rationale}` : null,
                                row.routing_retrieval_rationale ? `Retrieval: ${row.routing_retrieval_rationale}` : null,
                              ].filter(Boolean).join(' | ') || 'Routing details unavailable'}
                            >
                              details
                            </span>
                          </div>
                        </td>
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
                            <span className={`tag ${getLevelTagClass(row.confidence)}`} data-tooltip="Overall confidence level. High = passed all 5 verification checks: (1) answer matches question type, (2) explanation is grounded in image evidence, (3) confidence level is appropriate, (4) no contradictions with image, (5) format is correct. Medium/Low = some checks failed.">
                              confidence {(row.confidence || 'low').toLowerCase()}
                            </span>
                            <span className={`tag ${getLevelTagClass(row.self_consistency_level)}`} data-tooltip="Internal consistency of the answer. The system runs 5 self-checks. High = all 5 passed. Medium = 4 passed (1 minor issue). Low = 3 or fewer passed (significant issues). Measures whether the answer logic holds together.">
                              consistency {row.self_consistency_level}
                            </span>
                            <span className={`tag ${getLevelTagClass(row.cross_modal_agreement_level)}`} data-tooltip="Agreement between answer and image evidence. Two checks: (1) explanation grounded in actual image evidence, (2) no contradictions between answer and what the image shows. High = both checks passed. Medium = unclear/insufficient evidence. Low = contradicts image or makes unsupported claims.">
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
                        <td colSpan="9" style={{ textAlign: 'center', color: 'var(--color-text-muted)' }}>
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
