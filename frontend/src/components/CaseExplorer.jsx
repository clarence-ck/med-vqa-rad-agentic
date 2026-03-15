import { useMemo, useState } from 'react';
import AnswerCard from './AnswerCard';
import AgentTrace from './AgentTrace';
import ComparisonToggle from './ComparisonToggle';
import ExplanationCard from './ExplanationCard';
import VerificationPanel from './VerificationPanel';
import FailureModeDrawer from './FailureModeDrawer';

function getStatusTagClass(status) {
  if (status === 'verified') return 'tag-green';
  if (status === 'partially_verified') return 'tag-amber';
  return 'tag-red';
}

function getModeLabel(mode) {
  if (mode === 'baseline') return 'Single-shot Baseline';
  if (mode === 'workflow') return 'Workflow Baseline';
  return 'Agentic Pipeline';
}

function getSelectedResult(mode, analysisData) {
  if (mode === 'baseline') return analysisData?.baseline_result || null;
  if (mode === 'workflow') return analysisData?.workflow_result || null;
  return analysisData?.agentic_result || null;
}

function getSummaryBadge(mode, result) {
  if (!result) {
    return { className: 'tag-gray', label: 'pending' };
  }

  if (mode === 'agentic') {
    const status = result.verification_status || 'pending';
    return {
      className: getStatusTagClass(status),
      label: status.replace(/_/g, ' '),
    };
  }

  const confidence = result.confidence || 'low';
  return {
    className: confidence === 'high' ? 'tag-green' : confidence === 'medium' ? 'tag-amber' : 'tag-red',
    label: `${confidence} confidence`,
  };
}

export default function CaseExplorer({
  samples,
  loading,
  activeSplit,
  onSplitChange,
  testAccessEnabled,
  selectedSampleId,
  onSelectSample,
  analysisLoading,
  analysisData,
  onRefresh,
  onClearSelection,
}) {
  const [comparisonMode, setComparisonMode] = useState('agentic');
  const selectedResult = useMemo(
    () => getSelectedResult(comparisonMode, analysisData),
    [analysisData, comparisonMode],
  );
  const summaryBadge = useMemo(
    () => getSummaryBadge(comparisonMode, selectedResult),
    [comparisonMode, selectedResult],
  );

  const selectedSample = useMemo(
    () => samples.find((sample) => String(sample.id) === String(selectedSampleId)) || null,
    [samples, selectedSampleId],
  );

  const sampleSelectDisabled = Boolean((analysisData || analysisLoading) && selectedSampleId);

  const handleClear = () => {
    setComparisonMode('agentic');
    onClearSelection?.();
  };

  return (
    <main className="case-explorer-layout">
      <aside className="panel">
        <div className="panel-header">
          <h2><span className="icon">🗂️</span> Sample Cases</h2>
        </div>
        <div className="panel-body">
          <div className="form-group">
            <label className="form-label">Dataset split</label>
            <select
              className="form-select"
              value={activeSplit || 'all'}
              onChange={(e) => onSplitChange?.(e.target.value)}
            >
              <option value="train">Train — development only</option>
              <option value="test">Test — final evaluation only</option>
              <option value="all">All — mixed / not for reporting</option>
            </select>
            <div className="split-guidance-text">
              Case Explorer is mainly for debugging. Use <strong>train</strong> during development and avoid opening
              <strong> test</strong> unless you intentionally want to inspect held-out cases.
            </div>
            {activeSplit === 'test' && testAccessEnabled && (
              <div className="split-warning-banner split-warning-amber">
                <span>⚠️</span>
                <span>You are browsing test cases in a debugging view. Avoid tuning the system based on what you see here.</span>
              </div>
            )}
            {activeSplit === 'all' && (
              <div className="split-warning-banner">
                <span>ℹ️</span>
                <span>`All` mixes development and held-out cases. Use it only for exploration, not for final reporting.</span>
              </div>
            )}
          </div>
          <div className="form-group">
            <label className="form-label">Choose a dataset case</label>
            <select
              className="form-select"
              value={selectedSampleId ?? ''}
              onChange={(e) => onSelectSample?.(e.target.value)}
              disabled={sampleSelectDisabled}
            >
              <option value="">— Select a sample —</option>
              {samples.map((sample) => (
                <option key={sample.id} value={sample.id}>
                  [{sample.question_type?.replace(/_/g, ' ')}] {sample.question.slice(0, 70)}
                  {sample.question.length > 70 ? '...' : ''}
                </option>
              ))}
            </select>
            {sampleSelectDisabled && (
              <div className="split-guidance-text" style={{ marginTop: 'var(--space-2)' }}>
                Clear the current case before selecting another.
              </div>
            )}
          </div>

          {selectedSample && (
            <div className="card animate-fade-in">
              <div className="card-header">
                <span className="card-title">📌 Selected Case</span>
                <span className="tag tag-blue">ID {selectedSample.id}</span>
              </div>
              <div className="card-content">
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Question</span>
                  <span className="explorer-meta-value">{selectedSample.question}</span>
                </div>
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Ground truth</span>
                  <span className="explorer-meta-value">{selectedSample.answer}</span>
                </div>
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Question type</span>
                  <span className="tag tag-gray">{selectedSample.question_type?.replace(/_/g, ' ')}</span>
                </div>
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Split</span>
                  <span className="tag tag-blue">{selectedSample.split || 'unknown'}</span>
                </div>
              </div>
              <button className="btn btn-secondary btn-full" onClick={() => onRefresh?.()} disabled={analysisLoading}>
                {analysisLoading ? 'Loading case analysis...' : 'Run / Refresh Analysis'}
              </button>
              <button
                className="btn btn-ghost btn-full"
                type="button"
                onClick={handleClear}
                disabled={analysisLoading}
              >
                Clear selection
              </button>
            </div>
          )}

          {loading && !samples.length && (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <div className="loading-text">Loading sample cases...</div>
            </div>
          )}
        </div>
      </aside>

      <section className="panel" style={{ borderRight: '1px solid var(--color-border)' }}>
        <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2><span className="icon">🖼️</span> Case Preview</h2>
          {analysisData && (
            <ComparisonToggle mode={comparisonMode} onToggle={setComparisonMode} />
          )}
        </div>
        <div className="panel-body">
          {analysisData?.sample?.image_base64 ? (
            <div className="image-viewer">
              <img
                src={`data:${analysisData.sample.image_mime};base64,${analysisData.sample.image_base64}`}
                alt={analysisData.sample.image_name || 'Sample medical case'}
              />
            </div>
          ) : (
            <div className="empty-state">
              <span className="icon">🖼️</span>
              <div className="empty-state-title">No Case Loaded</div>
              <div className="empty-state-text">Select a dataset sample to inspect its image and outputs.</div>
            </div>
          )}

          {analysisData && (
            <div className="card animate-fade-in">
              <div className="card-header">
                <span className="card-title">📊 Case Summary</span>
                <span className={`tag ${summaryBadge.className}`}>
                  {summaryBadge.label}
                </span>
              </div>
              <div className="card-content">
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Selected view</span>
                  <span className="explorer-meta-value">{getModeLabel(comparisonMode)}</span>
                </div>
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Cached</span>
                  <span className="explorer-meta-value">{analysisData.cached ? 'Yes' : 'No'}</span>
                </div>
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Combined run time</span>
                  <span className="explorer-meta-value">{analysisData.duration_ms}ms</span>
                </div>
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Baseline answer</span>
                  <span className="explorer-meta-value">{analysisData.baseline_result?.answer || '—'}</span>
                </div>
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Workflow answer</span>
                  <span className="explorer-meta-value">{analysisData.workflow_result?.answer || '—'}</span>
                </div>
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">Agentic answer</span>
                  <span className="explorer-meta-value">{analysisData.agentic_result?.answer || '—'}</span>
                </div>
                <div className="explorer-meta-item">
                  <span className="explorer-meta-label">
                    {comparisonMode === 'agentic' ? 'Self-consistency' : 'Selected confidence'}
                  </span>
                  {comparisonMode === 'agentic' ? (
                    <span className="tag tag-gray">
                      {analysisData.agentic_result?.self_consistency_signal?.level || 'low'}
                    </span>
                  ) : (
                    <span className="tag tag-gray">
                      {selectedResult?.confidence || 'low'}
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <aside className="panel">
        <div className="panel-header">
          <h2><span className="icon">🧠</span> Case Analysis</h2>
        </div>
        <div className="panel-body">
          {!analysisData && !analysisLoading && (
            <div className="empty-state">
              <span className="icon">🔍</span>
              <div className="empty-state-title">No Case Analysis Yet</div>
              <div className="empty-state-text">Pick a case and run analysis to compare baseline and agentic outputs.</div>
            </div>
          )}

          {analysisLoading && (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <div className="loading-text">Running case analysis...</div>
            </div>
          )}

          {analysisData && comparisonMode === 'agentic' && (
            <>
              <AnswerCard result={analysisData.agentic_result} />
              <ExplanationCard result={analysisData.agentic_result} />
              <AgentTrace agentTrace={analysisData.agentic_result?.agent_trace} />
              <VerificationPanel result={analysisData.agentic_result} />
              <FailureModeDrawer result={analysisData.agentic_result} />
            </>
          )}

          {analysisData && comparisonMode === 'baseline' && (
            <div className="card animate-fade-in">
              <div className="card-header">
                <span className="card-title">⚡ Baseline (Single-shot) Result</span>
              </div>
              <div className="card-content">
                <p className="answer-text">{analysisData.baseline_result?.answer || 'No answer'}</p>
                {analysisData.baseline_result?.explanation && (
                  <p className="explanation-text" style={{ marginTop: 'var(--space-3)' }}>
                    {analysisData.baseline_result.explanation}
                  </p>
                )}
                {analysisData.baseline_result?.confidence && (
                  <div style={{ marginTop: 'var(--space-3)' }}>
                    <span className={`tag ${
                      analysisData.baseline_result.confidence === 'high' ? 'tag-green' :
                      analysisData.baseline_result.confidence === 'medium' ? 'tag-amber' : 'tag-red'
                    }`}>
                      {analysisData.baseline_result.confidence} confidence
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {analysisData && comparisonMode === 'workflow' && (
            <div className="card animate-fade-in">
              <div className="card-header">
                <span className="card-title">🧭 Workflow Baseline Result</span>
              </div>
              <div className="card-content">
                <p className="answer-text">{analysisData.workflow_result?.answer || 'No answer'}</p>
                {analysisData.workflow_result?.explanation && (
                  <p className="explanation-text" style={{ marginTop: 'var(--space-3)' }}>
                    {analysisData.workflow_result.explanation}
                  </p>
                )}
                {analysisData.workflow_result?.confidence && (
                  <div style={{ marginTop: 'var(--space-3)' }}>
                    <span className={`tag ${
                      analysisData.workflow_result.confidence === 'high' ? 'tag-green' :
                      analysisData.workflow_result.confidence === 'medium' ? 'tag-amber' : 'tag-red'
                    }`}>
                      {analysisData.workflow_result.confidence} confidence
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </aside>
    </main>
  );
}
