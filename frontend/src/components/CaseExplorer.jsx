import { useMemo, useState } from 'react';
import AnswerCard from './AnswerCard';
import AgentTrace from './AgentTrace';
import ComparisonToggle from './ComparisonToggle';
import ExplanationCard from './ExplanationCard';
import VerificationPanel from './VerificationPanel';
import FailureModeDrawer from './FailureModeDrawer';

function fmtMs(ms) {
  if (ms == null) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
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

function getStatusTagClass(status) {
  if (status === 'verified') return 'tag-green';
  if (status === 'partially_verified') return 'tag-amber';
  return 'tag-red';
}

function getKnowledgeSupportDetails(agenticResult) {
  const trace = agenticResult?.agent_trace || [];
  const knowledgeStep = trace.find(
    (step) => String(step?.agent || '').toLowerCase() === 'knowledge support',
  );

  if (!knowledgeStep) {
    return {
      triggered: false,
      status: 'unknown',
      keyFactsCount: 0,
      summary: '',
    };
  }

  const status = String(knowledgeStep.status || 'unknown').toLowerCase();
  const keyFacts = knowledgeStep.output?.key_facts || [];

  return {
    triggered: status !== 'skipped',
    status,
    keyFactsCount: keyFacts.length,
    summary: knowledgeStep.summary || '',
  };
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
  if (mode === 'baseline') return { className: 'tag-amber', label: 'Baseline' };
  if (mode === 'workflow') return { className: 'tag-blue', label: 'Workflow' };
  return { className: 'tag-green', label: 'Agentic' };
}

function getConfidenceTagClass(confidence) {
  const normalized = (confidence || 'low').toLowerCase();
  return normalized === 'high' ? 'tag-green' : normalized === 'medium' ? 'tag-amber' : 'tag-red';
}

function getVerificationTagClass(status) {
  const normalized = (status || 'unknown').toLowerCase();
  if (normalized === 'verified') return 'tag-green';
  if (normalized === 'partially_verified') return 'tag-amber';
  return 'tag-gray';
}

function MethodologyLabels() {
  return (
    <div className="methodology-compact" style={{ 
      marginBottom: 'var(--space-4)', 
      paddingBottom: 'var(--space-4)', 
      borderBottom: '1px solid var(--color-border)', 
      fontSize: 'var(--font-size-xs)',
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gap: 'var(--space-4)'
    }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}>
          <span style={{ color: 'var(--color-accent)', fontWeight: 'bold' }}>🔗 Agentic Pipeline</span>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.4' }}>
          Decomposes queries into visual interpretation, context retrieval, and iterative reasoning with a final verification loop.
        </p>
      </div>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}>
          <span style={{ color: 'var(--color-warning)', fontWeight: 'bold' }}>🧭 Workflow Baseline</span>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.4' }}>
          A deterministic sequence of analysis steps that follows a fixed path without agentic self-correction or feedback.
        </p>
      </div>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}>
          <span style={{ color: 'var(--color-error)', fontWeight: 'bold' }}>⚡ Baseline (Single-shot)</span>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.4' }}>
          A direct, single-prompt approach where the LLM answers from the image without any intermediate decomposition.
        </p>
      </div>
    </div>
  );
}

export default function CaseExplorer({
  samples,
  loading,
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
  const knowledgeSupport = useMemo(
    () => getKnowledgeSupportDetails(analysisData?.agentic_result),
    [analysisData],
  );

  const selectedSample = useMemo(
    () => samples.find((sample) => String(sample.id) === String(selectedSampleId)) || null,
    [samples, selectedSampleId],
  );

  const selectedConfidence = selectedResult?.confidence || 'low';
  const selectedExplanation = selectedResult?.explanation || '';
  const overallVerification = analysisData?.agentic_result?.verification_status || 'unknown';
  const hasRetrieverDetails = knowledgeSupport.triggered;
  const selfConsistencyLevel = analysisData?.agentic_result?.self_consistency_signal?.level || 'low';
  const showAgenticDetails = comparisonMode === 'agentic';
  const questionAnalysis = analysisData?.agentic_result?.question_analysis || {};
  const questionType = questionAnalysis.question_type?.replace(/_/g, ' ') || '—';
  const questionClarity = questionAnalysis.question_clarity || 'unknown';
  const reasoningType = questionAnalysis.reasoning_type?.replace(/_/g, ' ') || '—';
  const expectedAnswerFormat = questionAnalysis.expected_answer_format || '—';
  const reviewRecommended = Boolean(analysisData?.agentic_result?.should_flag_for_review);
  const issueCount = analysisData?.agentic_result?.issues_found?.length || 0;
  const crossModalAgreement = analysisData?.agentic_result?.cross_modal_agreement?.level || 'unknown';

  const groupedSamples = useMemo(() => {
    const groups = samples.reduce((acc, sample) => {
      const key = sample.question_type || 'other';
      if (!acc[key]) acc[key] = [];
      acc[key].push(sample);
      return acc;
    }, {});

    Object.values(groups).forEach((group) => {
      group.sort((a, b) => (a.question || '').localeCompare(b.question || ''));
    });

    const orderedKeys = [
      ...QUESTION_TYPE_ORDER.filter((key) => groups[key]?.length),
      ...Object.keys(groups)
        .filter((key) => !QUESTION_TYPE_ORDER.includes(key))
        .sort(),
    ];

    return orderedKeys.map((key) => ({
      key,
      label: key.replace(/_/g, ' '),
      items: groups[key],
    }));
  }, [samples]);

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
            <label className="form-label">Choose a dataset case</label>
            <div className="split-guidance-text" style={{ marginBottom: 'var(--space-2)' }}>
              Dataset split: test only
            </div>
            <select
              className="form-select"
              value={selectedSampleId ?? ''}
              onChange={(e) => onSelectSample?.(e.target.value)}
              disabled={sampleSelectDisabled}
            >
              <option value="">— Select a sample —</option>
              {groupedSamples.map((group) => (
                <optgroup key={group.key} label={group.label}>
                  {group.items.map((sample) => (
                    <option key={sample.id} value={sample.id}>
                      [{sample.id}] {sample.question.slice(0, 62)}
                      {sample.question.length > 70 ? '...' : ''}
                    </option>
                  ))}
                </optgroup>
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
              </div>
              <button
                className="btn btn-secondary btn-full"
                type="button"
                onClick={onRefresh}
                disabled={analysisLoading}
                style={{ marginTop: 'var(--space-4)' }}
              >
                Refresh analysis
              </button>
              <button
                className="btn btn-ghost btn-full"
                type="button"
                onClick={handleClear}
                disabled={analysisLoading}
                style={{ marginTop: 'var(--space-4)' }}
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
            <ComparisonToggle
            mode={comparisonMode}
            onToggle={setComparisonMode}
            agenticMs={analysisData.agentic_duration_ms ?? analysisData.agentic_result?.total_duration_ms}
            workflowMs={analysisData.workflow_duration_ms}
            baselineMs={analysisData.baseline_duration_ms}
          />
          )}
        </div>
        <div className="panel-body">
          <MethodologyLabels />
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
            <div className="card animate-fade-in case-summary-card">
              <div className="card-header">
                <span className="card-title">📊 Case Summary</span>
                <div className="case-summary-header-tags">
                  <span className={`tag ${summaryBadge.className}`}>
                    {summaryBadge.label}
                  </span>
                </div>
              </div>

              <div className="card-content case-summary-content">
                <section className="summary-hero">
                  <div className="summary-hero-meta">
                    <span className="summary-hero-label">Current view</span>
                    <span className="summary-hero-view">{getModeLabel(comparisonMode)}</span>
                  </div>
                  <div className="summary-hero-answer">
                    {selectedResult?.answer || 'No answer available'}
                  </div>
                  <div className="summary-hero-footer">
                    <span className={`tag ${getConfidenceTagClass(selectedConfidence)}`}>
                      {selectedConfidence} confidence
                    </span>
                    {showAgenticDetails && (
                      <>
                        <span className={`tag ${getVerificationTagClass(overallVerification)}`}>
                          {overallVerification.replace(/_/g, ' ')}
                        </span>
                        <span className={`tag ${getConfidenceTagClass(selfConsistencyLevel)}`}>
                          consistency {selfConsistencyLevel}
                        </span>
                        <span className={`tag ${getConfidenceTagClass(crossModalAgreement)}`}>
                          cross-modal {crossModalAgreement}
                        </span>
                      </>
                    )}
                  </div>
                </section>

                <div className={`summary-section-grid ${showAgenticDetails ? 'summary-section-grid-agentic' : 'summary-section-grid-baseline'}`}>
                  <section className="summary-section">
                    <div className="summary-section-title">Overview</div>
                    <div className="summary-stat-list">
                      <div className="summary-stat-row">
                        <span className="summary-stat-label">Selected sample</span>
                        <span className="summary-stat-value">ID {selectedSample?.id ?? '—'}</span>
                      </div>
                      <div className="summary-stat-row">
                        <span className="summary-stat-label">Selected view</span>
                        <span className="summary-stat-value">{getModeLabel(comparisonMode)}</span>
                      </div>
                      <div className="summary-stat-row">
                        <span className="summary-stat-label">Duration</span>
                        <span className="summary-stat-value">
                          {fmtMs(
                            comparisonMode === 'agentic'
                              ? (analysisData?.agentic_duration_ms ?? analysisData?.agentic_result?.total_duration_ms)
                              : comparisonMode === 'workflow'
                                ? analysisData?.workflow_duration_ms
                                : analysisData?.baseline_duration_ms
                          )}
                        </span>
                      </div>

                      <div className="summary-summary-text">
                        {showAgenticDetails
                          ? 'Agentic mode includes question understanding, retrieval, and verification metadata.'
                          : 'This view shows the selected baseline answer path for direct comparison.'}
                      </div>
                    </div>
                  </section>

                  {showAgenticDetails ? (
                    <>
                      <section className="summary-section">
                        <div className="summary-section-title">Agentic Quality</div>
                        <div className="summary-stat-list">
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="The model's self-reported confidence in its final answer, based on the currently selected methodology (Agentic, Workflow, or Baseline).">Selected confidence</span>
                            <span className={`tag ${getConfidenceTagClass(selectedConfidence)}`}>
                              {selectedConfidence}
                            </span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Based on the Verification Agent's checks. 'Verified' means it passed all checks (question matched, evidence grounded, strength appropriate, no contradictions, format correct). 'Partially_verified' means it passed most but not all checks.">Verification</span>
                            <span className={`tag ${getVerificationTagClass(overallVerification)}`}>
                              {overallVerification.replace(/_/g, ' ')}
                            </span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Derived from the proportion of verification checks passed. 'High' means all checks passed; 'medium' means at most 1 check failed; 'low' otherwise.">Self-consistency</span>
                            <span className={`tag ${getConfidenceTagClass(selfConsistencyLevel)}`}>
                              {selfConsistencyLevel}
                            </span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Evaluates whether the answer is grounded in the visual evidence AND free of contradictions with the image interpretation. 'High' = both true; 'low' = either false; 'medium' otherwise.">Cross-modal agreement</span>
                            <span className={`tag ${getConfidenceTagClass(crossModalAgreement)}`}>{crossModalAgreement}</span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Set to 'yes' if the pipeline encountered 'unsupported claims', 'visual contradictions', a 'parser failure', or if the Verification Agent explicitly recommended flagging for review.">Review recommended</span>
                            <span className={`tag ${reviewRecommended ? 'tag-red' : 'tag-green'}`}>
                              {reviewRecommended ? 'yes' : 'no'}
                            </span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Number of explicit flaws identified by the Verification Agent during the final review phase.">Issues found</span>
                            <span className={`tag ${issueCount === 0 ? 'tag-green' : 'tag-amber'}`}>{issueCount}</span>
                          </div>
                        </div>
                      </section>

                      <section className="summary-section">
                        <div className="summary-section-title">Retrieval</div>
                        <div className="summary-stat-list">
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Whether the Knowledge Retriever agent was invoked. It is only triggered for abnormality descriptions, explanation requests, or when medical knowledge is explicitly required.">Knowledge retriever</span>
                            <span className={`tag ${hasRetrieverDetails ? 'tag-green' : 'tag-gray'}`}>
                              {hasRetrieverDetails ? 'triggered' : 'skipped'}
                            </span>
                          </div>
                          <div className="summary-stat-row">
                            <span className="summary-stat-label" data-tooltip="Number of specific 'key facts' extracted by the Knowledge Support agent from external context to assist the reasoning phase.">Key facts</span>
                            <span className="summary-stat-value">{knowledgeSupport.keyFactsCount}</span>
                          </div>
                          <div className="summary-summary-text">
                            {hasRetrieverDetails ? knowledgeSupport.summary : 'No retrieval context was added for this case.'}
                          </div>
                        </div>
                      </section>
                    </>
                  ) : (
                    <section className="summary-section summary-section-wide">
                      <div className="summary-section-title">Selected Result</div>
                      <div className="summary-stat-list">
                        <div className="summary-stat-row summary-stat-row-inline">
                          <span className="summary-stat-label" data-tooltip="The model's self-reported confidence in its final answer, based on the currently selected methodology (Agentic, Workflow, or Baseline).">Selected confidence</span>
                          <span className={`tag ${getConfidenceTagClass(selectedConfidence)}`}>
                            {selectedConfidence}
                          </span>
                        </div>
                        <div className="summary-stat-row summary-stat-row-inline">
                          <span className="summary-stat-label">Result type</span>
                          <span className="summary-stat-value">{comparisonMode === 'workflow' ? 'workflow baseline' : 'single-shot baseline'}</span>
                        </div>
                        <div className="summary-summary-text">
                          {selectedExplanation || 'No explanation was returned for this result.'}
                        </div>
                      </div>
                    </section>
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

          {analysisData && comparisonMode !== 'agentic' && selectedResult && (
            <>
              <div className="card animate-fade-in">
                <div className="card-header">
                  <span className="card-title">💡 Answer</span>
                  <span className={`tag ${getConfidenceTagClass(selectedResult.confidence)}`}>
                    {(selectedResult.confidence || 'low')} confidence
                  </span>
                </div>
                <div className="card-content">
                  <p className="answer-text">{selectedResult.answer || 'No answer available.'}</p>
                </div>
              </div>

              {selectedResult.explanation && (
                <div className="card animate-fade-in" style={{ animationDelay: '0.1s' }}>
                  <div className="card-header">
                    <span className="card-title">📋 Explanation</span>
                  </div>
                  <div className="card-content">
                    <p className="explanation-text">{selectedResult.explanation}</p>
                  </div>
                </div>
              )}

              <div className="card animate-fade-in" style={{ animationDelay: '0.15s' }}>
                <div className="card-header">
                  <span className="card-title">ℹ️ About This Mode</span>
                </div>
                <div className="card-content">
                  <p className="explanation-text">
                    {comparisonMode === 'workflow'
                      ? 'Workflow Baseline: a deterministic sequence of analysis steps without agentic self-correction.'
                      : 'Single-shot Baseline: a direct, single-prompt approach without intermediate decomposition.'}
                    {' '}Switch to Agentic Pipeline to inspect step-by-step reasoning, retrieval, and verification details.
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      </aside>
    </main>
  );
}
