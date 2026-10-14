 import { useEffect, useMemo, useRef, useState } from 'react';
 
function formatPercent1(value) {
  return `${((value || 0) * 100).toFixed(1)}%`;
}

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

function getControllerPathTagClass(path) {
  if (path === 'direct') return 'tag-green';
  if (path === 'relook' || path === 'reanswer') return 'tag-amber';
  if (path === 'abstained') return 'tag-red';
  return 'tag-gray';
}

function getBreakdownTooltip(mode) {
  if (mode === 'controller_path') {
    return 'Breakdown by what the agentic controller did: direct (look → answer → check → finish), relook (a focused re-examination of a region), reanswer (answer revised after a failed check) or abstained. The pipeline column is the fixed six-stage ablation on the same cases.';
  }

  return 'Breakdown by the dataset question type (independent of the agent’s own classification), using the same scoring rules as the summary above.';
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

function BreakdownAccuracyCell({ value, color }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
      <span style={{ fontSize: '11px', fontWeight: 600 }}>{formatPercent(value)}</span>
      <div style={{ width: '40px', backgroundColor: 'var(--color-border)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
        <div style={{ width: `${value * 100}%`, height: '100%', backgroundColor: color, borderRadius: '3px' }}></div>
      </div>
    </div>
  );
}

function JudgeTag({ verdict }) {
  if (verdict !== 'correct') return null;
  return (
    <span
      className="tag tag-green"
      style={{ fontSize: '9px', padding: '1px 5px', marginLeft: '4px' }}
      data-tooltip="The strict scorer rejected this wording, but the blinded LLM judge rated it as meaning the same as the ground truth; it counts toward LLM-judged accuracy."
    >
      LLM-judged ✓
    </span>
  );
}

function formatLabel(value) {
  return String(value || 'unknown').replace(/_/g, ' ');
}

function SystemPerformanceCard({ title, icon, accuracy, judgedAccuracy, avgDuration, medianDuration }) {
  return (
    <div className="card evaluation-system-card">
      <div className="card-header">
        <span className="card-title">{icon} {title}</span>
      </div>
      <div className="evaluation-system-stats">
        <div className="evaluation-system-stat">
          <div className="evaluation-list-subtitle">Overall accuracy (strict scorer)</div>
          <div className="evaluation-metric-value">{formatPercent1(accuracy)}</div>
          {judgedAccuracy !== null && judgedAccuracy !== undefined && (
            <div
              className="evaluation-list-subtitle"
              data-tooltip="Upper bound: open-ended answers the strict scorer rejected but a blinded LLM judge (gpt-6.1-sol) rated as meaning the same as the ground truth also count as correct."
              data-tooltip-align="left"
            >
              LLM-judged: {formatPercent1(judgedAccuracy)}
            </div>
          )}
        </div>
        <div className="evaluation-system-stat">
          <div
            className="evaluation-list-subtitle"
            data-tooltip="Mean latency per case, measured while several cases ran concurrently."
            data-tooltip-align="left"
          >
            Mean time / case
          </div>
          <div className="evaluation-system-runtime">{fmtMs(avgDuration)}</div>
        </div>
        <div className="evaluation-system-stat">
          <div
            className="evaluation-list-subtitle"
            data-tooltip="Median latency per case: the typical case. The mean is pulled up by a few long cases (e.g. focused re-looks)."
            data-tooltip-align="left"
          >
            Median time / case
          </div>
          <div className="evaluation-system-runtime">{fmtMs(medianDuration)}</div>
        </div>
      </div>
    </div>
  );
}

function AccuracyMetricStack({ baseline, workflow, agentic, pipeline }) {
  return (
    <div className="evaluation-metric-stack">
      <span className="tag tag-amber">Baseline {formatPercent1(baseline)}</span>
      <span className="tag tag-blue">Workflow {formatPercent1(workflow)}</span>
      <span className="tag tag-green">Agentic {formatPercent1(agentic)}</span>
      {pipeline !== null && pipeline !== undefined && (
        <span className="tag tag-gray">Pipeline (ablation) {formatPercent1(pipeline)}</span>
      )}
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
  const rows = useMemo(() => evaluationData?.rows || [], [evaluationData]);
  const [selectedSampleIds, setSelectedSampleIds] = useState([]);
  const [isSampleDropdownOpen, setIsSampleDropdownOpen] = useState(false);
  const [breakdownMode, setBreakdownMode] = useState('question_type');
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
  const hasPipeline = Boolean(summary?.pipeline_cases);
  const isControllerMode = breakdownMode === 'controller_path';
  const breakdownRows = (isControllerMode ? summary?.by_controller_path : summary?.by_question_type) || [];
  const breakdownTitle = isControllerMode ? 'Accuracy by Controller Path' : 'Accuracy by Question Type';
  const breakdownPrimaryLabel = isControllerMode ? 'Controller Path' : 'Question Type';
  const breakdownContextTag = isControllerMode ? 'Agentic decisions' : 'Dataset labels';
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

  return (
    <main className="evaluation-layout">
      <section className="evaluation-header-strip">
        <div>
          <h1 className="evaluation-title">Evaluation Dashboard</h1>
          <p className="evaluation-subtitle">Accuracy over cached Case Explorer runs, scored by backend/utils/scoring.py. Close-ended: yes/no answers must start with the correct word; binary terms (normal/abnormal, present/absent, …) must name the correct term and not its opposite. Counting: the numbers must match. Open-ended: every content word of the ground truth must appear (aliases and plurals folded), or at least 60% of them when the ground truth is longer than 3 words; left/right conflicts, and answers starting with “no” when the ground truth does not, are wrong; S-PubMedBert-MS-MARCO similarity is only a strict fallback (≥ 0.95 short, ≥ 0.90 long).</p>
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
              judgedAccuracy={summary.baseline_judged_accuracy}
              icon="⚡"
              accuracy={summary.baseline_accuracy}
              avgDuration={summary.avg_baseline_duration_ms}
              medianDuration={summary.median_baseline_duration_ms}
            />
            <SystemPerformanceCard
              title="Workflow"
              judgedAccuracy={summary.workflow_judged_accuracy}
              icon="🧭"
              accuracy={summary.workflow_accuracy}
              avgDuration={summary.avg_workflow_duration_ms}
              medianDuration={summary.median_workflow_duration_ms}
            />
            <SystemPerformanceCard
              title="Agentic (controller loop)"
              judgedAccuracy={summary.agentic_judged_accuracy}
              icon="🔗"
              accuracy={summary.agentic_accuracy}
              avgDuration={summary.avg_agentic_duration_ms}
              medianDuration={summary.median_agentic_duration_ms}
            />
            {hasPipeline && (
              <SystemPerformanceCard
                title={`Fixed Pipeline — ablation (${summary.pipeline_cases} cases)`}
                icon="🧱"
                accuracy={summary.pipeline_accuracy}
                judgedAccuracy={summary.pipeline_judged_accuracy}
                avgDuration={summary.avg_pipeline_duration_ms}
                medianDuration={summary.median_pipeline_duration_ms}
              />
            )}
          </section>

          <section className="evaluation-metric-grid animate-fade-in">
            <div className="card">
              <div className="card-header"><span className="card-title">Total Cases</span></div>
              <div className="evaluation-metric-value">{summary.total_cases}</div>
            </div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">3-System Disagreement</span>
                <span className="tag tag-gray" data-tooltip="Percentage of cases where at least one pair of system answers does not agree under the same matching rules used for accuracy." data-tooltip-align="left">info</span>
              </div>
              <div className="evaluation-metric-value">{formatPercent(summary.disagreement_rate)}</div>
            </div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">Lexical Overlap</span>
                <span className="tag tag-gray" data-tooltip="Average token F1 overlap score (0-1) across all cases. This is a surface-level diagnostic similarity metric only and is not used for correctness. Compare with Open-ended Accuracy, which uses content-word matching with a strict embedding fallback." data-tooltip-align="left">info</span>
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
                  <span className="tag tag-gray" data-tooltip="Yes/no ground truth: the answer must start with the correct word (“No, there is…” counts as no). Binary ground truth (normal/abnormal, present/absent, positive/negative, benign/malignant, acute/chronic, diffuse/focal): the answer must contain the correct term and not its opposite." data-tooltip-align="left">info</span>
                </div>
              </div>
              <AccuracyMetricStack
                baseline={summary.baseline_close_ended_accuracy}
                workflow={summary.workflow_close_ended_accuracy}
                agentic={summary.agentic_close_ended_accuracy}
                pipeline={hasPipeline ? summary.pipeline_close_ended_accuracy : null}
              />
            </div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">Open-ended Accuracy</span>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className="tag tag-gray">{summary.open_ended_cases} cases</span>
                  <span className="tag tag-gray" data-tooltip="Every content word of the ground truth must appear in the answer (aliases such as CT = computed tomography; plurals folded), or at least 60% of them when the ground truth is longer than 3 words. Answers naming the opposite side (left vs right), or starting with “no” when the ground truth does not, are wrong. S-PubMedBert-MS-MARCO similarity is only a strict fallback (≥ 0.95 short, ≥ 0.90 long). Counting questions compare the numbers." data-tooltip-align="left">info</span>
                </div>
              </div>
              <AccuracyMetricStack
                baseline={summary.baseline_open_ended_accuracy}
                workflow={summary.workflow_open_ended_accuracy}
                agentic={summary.agentic_open_ended_accuracy}
                pipeline={hasPipeline ? summary.pipeline_open_ended_accuracy : null}
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
                    <option value="controller_path">Controller Path</option>
                  </select>
                  <span className={`tag ${isControllerMode ? 'tag-green' : 'tag-gray'}`}>{breakdownContextTag}</span>
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
                      {hasPipeline && <th>Pipeline (ablation)</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {breakdownRows.map((item) => (
                      <tr key={item.question_type || item.label}>
                        <td style={{ fontWeight: 500, color: 'var(--color-text-primary)' }}>
                          {formatLabel(item.question_type || item.label)}
                        </td>
                        <td><span className="tag tag-gray">{item.count}</span></td>
                        <td><BreakdownAccuracyCell value={item.baseline_accuracy} color="var(--color-warning)" /></td>
                        <td><BreakdownAccuracyCell value={item.workflow_accuracy} color="var(--color-accent)" /></td>
                        <td><BreakdownAccuracyCell value={item.agentic_accuracy} color="var(--color-success)" /></td>
                        {hasPipeline && (
                          <td>
                            {item.pipeline_accuracy === null || item.pipeline_accuracy === undefined
                              ? <span className="tag tag-gray">—</span>
                              : <BreakdownAccuracyCell value={item.pipeline_accuracy} color="var(--color-text-muted)" />}
                          </td>
                        )}
                      </tr>
                    ))}
                    {breakdownRows.length === 0 && (
                      <tr>
                        <td colSpan={hasPipeline ? 6 : 5} style={{ textAlign: 'center', color: 'var(--color-text-muted)' }}>
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
                      <th>Controller</th>
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
                              <span style={{ color: row.baseline_correct ? 'var(--color-success)' : 'inherit', textDecoration: !row.baseline_correct && row.baseline_answer ? 'line-through' : 'none' }}>{row.baseline_answer || '—'}</span>{' '}<JudgeTag verdict={row.baseline_judge_verdict} />
                            </div>
                            <div>
                              <span className="tag tag-blue" style={{ fontSize: '9px', padding: '1px 5px', minWidth: '55px', justifyContent: 'center', marginRight: '6px' }}>Workflow</span>
                              <span style={{ color: row.workflow_correct ? 'var(--color-success)' : 'inherit', textDecoration: !row.workflow_correct && row.workflow_answer ? 'line-through' : 'none' }}>{row.workflow_answer || '—'}</span>{' '}<JudgeTag verdict={row.workflow_judge_verdict} />
                            </div>
                            <div>
                              <span className="tag tag-green" style={{ fontSize: '9px', padding: '1px 5px', minWidth: '55px', justifyContent: 'center', marginRight: '6px' }}>Agentic</span>
                              <span style={{ color: row.agentic_correct ? 'var(--color-success)' : 'inherit', textDecoration: !row.agentic_correct && row.agentic_answer ? 'line-through' : 'none' }}>{row.agentic_answer || '—'}</span>{' '}<JudgeTag verdict={row.agentic_judge_verdict} />
                            </div>
                            {row.pipeline_answer !== null && row.pipeline_answer !== undefined && (
                              <div>
                                <span className="tag tag-gray" style={{ fontSize: '9px', padding: '1px 5px', minWidth: '55px', justifyContent: 'center', marginRight: '6px' }}>Pipeline</span>
                                <span style={{ color: row.pipeline_correct ? 'var(--color-success)' : 'inherit', textDecoration: !row.pipeline_correct && row.pipeline_answer ? 'line-through' : 'none' }}>{row.pipeline_answer || '—'}</span>{' '}<JudgeTag verdict={row.pipeline_judge_verdict} />
                              </div>
                            )}
                          </div>
                        </td>
                        <td><span className={`tag ${getWinnerTagClass(row.winner)}`}>{row.winner.replace(/_/g, ' ')}</span></td>
                        <td>
                          <div className="evaluation-signal-stack">
                            <span
                              className={`tag ${getControllerPathTagClass(row.controller_path)}`}
                              data-tooltip={(row.controller_actions || []).join(' → ') || 'No controller actions recorded'}
                              data-tooltip-align="left"
                            >
                              {formatLabel(row.controller_path)}
                            </span>
                            {row.agentic_attempts ? (
                              <span className="evaluation-list-subtitle">{row.agentic_attempts} answer{row.agentic_attempts === 1 ? '' : 's'}</span>
                            ) : null}
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
                            <span className={`tag ${getLevelTagClass(row.self_consistency_level)}`} data-tooltip="Self-consistency: the reasoner is re-asked independently (default 2 times) and the re-answers are compared with the final answer. High = all agree; medium = at least half agree; low = most disagree (the case is flagged for review).">
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
