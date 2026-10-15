import { useEffect, useMemo, useRef, useState } from 'react';
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

function getModeLabel(mode) {
  if (mode === 'baseline') return 'Single-shot Baseline';
  if (mode === 'workflow') return 'Workflow Baseline';
  return 'Agentic Loop';
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

function getControllerPathTagClass(path) {
  if (path === 'direct') return 'tag-green';
  if (path === 'relook' || path === 'reanswer') return 'tag-amber';
  if (path === 'abstained') return 'tag-red';
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
          <span style={{ color: 'var(--color-accent)', fontWeight: 'bold' }}>🔗 Agentic Loop</span>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.4' }}>
          A controller agent looks at the image with the question in mind, answers, checks the answer against the image, and decides whether to re-examine a region, revise, abstain or finish.
        </p>
      </div>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}>
          <span style={{ color: 'var(--color-warning)', fontWeight: 'bold' }}>🧭 Workflow Baseline</span>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.4' }}>
          A fixed sequence of steps (a question-agnostic image description, question analysis, then hand-written rules per question type) without self-correction or feedback.
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
  samplesError,
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

  const selectedConfidence = selectedResult?.confidence || 'low';
  const selectedExplanation = selectedResult?.explanation || '';
  const overallVerification = analysisData?.agentic_result?.verification_status || 'unknown';
  const selfConsistencyLevel = analysisData?.agentic_result?.self_consistency_signal?.level || 'low';
  const showAgenticDetails = comparisonMode === 'agentic';
  const reviewRecommended = Boolean(analysisData?.agentic_result?.should_flag_for_review);
  const issueCount = analysisData?.agentic_result?.issues_found?.length || 0;
  const crossModalAgreement = analysisData?.agentic_result?.cross_modal_agreement?.level || 'unknown';
  const agentic = analysisData?.agentic_result || {};
  const controllerPath = agentic.controller_path || (agentic.agent_mode === 'pipeline' ? 'pipeline' : 'unknown');
  const controllerActions = agentic.controller_actions || [];
  const answerAttempts = agentic.attempts || 0;
  const selectedAttempt = agentic.selected_attempt || 0;

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

  const [searchQuery, setSearchQuery] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const dropdownRef = useRef(null);
  const inputRef = useRef(null);

  const flattenedSamples = useMemo(() => {
    return groupedSamples.flatMap((group) =>
      group.items.map((sample) => ({
        ...sample,
        groupLabel: group.label,
      })),
    );
  }, [groupedSamples]);

  const filteredSamples = useMemo(() => {
    if (!searchQuery.trim()) return flattenedSamples;
    const query = searchQuery.trim().toLowerCase();
    const matches = flattenedSamples.filter(
      (sample) =>
        String(sample.id).toLowerCase().includes(query) ||
        sample.question.toLowerCase().includes(query),
    );
    // An exact ID match comes first (so "457" selects case 457, not 1457), in its own group.
    const exact = matches.find((sample) => String(sample.id) === query);
    if (!exact) return matches;
    return [{ ...exact, groupLabel: 'Matching ID' }, ...matches.filter((sample) => sample !== exact)];
  }, [flattenedSamples, searchQuery]);

  const filteredGroups = useMemo(() => {
    const groups = {};
    filteredSamples.forEach((sample) => {
      if (!groups[sample.groupLabel]) {
        groups[sample.groupLabel] = [];
      }
      groups[sample.groupLabel].push(sample);
    });
    return Object.entries(groups).map(([label, items]) => ({
      label,
      items,
    }));
  }, [filteredSamples]);

  const selectedSampleDisplay = useMemo(() => {
    if (!selectedSampleId) return '';
    const sample = flattenedSamples.find((s) => String(s.id) === String(selectedSampleId));
    return sample ? `[${sample.id}] ${sample.question.slice(0, 50)}${sample.question.length > 50 ? '...' : ''}` : '';
  }, [flattenedSamples, selectedSampleId]);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (isDropdownOpen) {
      inputRef.current?.focus();
    }
  }, [isDropdownOpen]);

  const handleKeyDown = (e) => {
    if (!isDropdownOpen) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightedIndex((prev) => Math.min(prev + 1, filteredSamples.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const sample = filteredSamples[highlightedIndex];
      if (sample) {
        onSelectSample?.(sample.id);
        setIsDropdownOpen(false);
        setSearchQuery('');
      }
    } else if (e.key === 'Escape') {
      setIsDropdownOpen(false);
    }
  };

  const handleSelect = (sampleId) => {
    onSelectSample?.(sampleId);
    setIsDropdownOpen(false);
    setSearchQuery('');
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
              The evaluation questions (experiments/eval_set.json)
            </div>
            <div ref={dropdownRef} style={{ position: 'relative' }}>
              <button
                type="button"
                className="form-select"
                onClick={() => {
                  if (sampleSelectDisabled) return;
                  if (!isDropdownOpen) setHighlightedIndex(0);
                  setIsDropdownOpen(!isDropdownOpen);
                }}
                disabled={sampleSelectDisabled}
                style={{
                  textAlign: 'left',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {selectedSampleDisplay || '— Select a sample —'}
                </span>
                <span style={{ marginLeft: '8px' }}>{isDropdownOpen ? '▲' : '▼'}</span>
              </button>
              {sampleSelectDisabled && (
                <div className="split-guidance-text" style={{ marginTop: 'var(--space-2)' }}>
                  Clear the current case before selecting another.
                </div>
              )}
              {isDropdownOpen && (
                <div
                  style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    right: 0,
                    background: 'var(--color-surface, #fff)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius, 4px)',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                    zIndex: 100,
                    maxHeight: '320px',
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                >
                  <div style={{ padding: '8px', borderBottom: '1px solid var(--color-border)' }}>
                    <input
                      ref={inputRef}
                      type="text"
                      value={searchQuery}
                      onChange={(e) => {
                        setSearchQuery(e.target.value);
                        setHighlightedIndex(0);
                      }}
                      onKeyDown={handleKeyDown}
                      placeholder="Search by ID or question..."
                      style={{
                        width: '100%',
                        padding: '6px 10px',
                        border: '1px solid var(--color-border)',
                        borderRadius: '4px',
                        fontSize: 'var(--font-size-sm)',
                      }}
                    />
                  </div>
                  <div style={{ overflowY: 'auto', flex: 1, maxHeight: '240px' }}>
                    {filteredGroups.length === 0 ? (
                      <div style={{ padding: '12px', color: 'var(--color-text-secondary)', textAlign: 'center' }}>
                        {samplesError ? `Could not load cases: ${samplesError}` : 'No samples found'}
                      </div>
                    ) : (
                      filteredGroups.map((group, groupIdx) => (
                        <div key={group.label} style={{ marginTop: groupIdx > 0 ? '8px' : 0 }}>
                          <div
                            style={{
                              padding: '8px 12px',
                              fontSize: 'var(--font-size-xs)',
                              fontWeight: 700,
                              color: 'var(--color-accent, #0066cc)',
                              background: 'linear-gradient(to right, rgba(0,102,204,0.08), rgba(0,102,204,0.02))',
                              borderLeft: '3px solid var(--color-accent, #0066cc)',
                              borderBottom: '1px solid var(--color-border)',
                              textTransform: 'uppercase',
                              letterSpacing: '1px',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                            }}
                          >
                            <span style={{ fontSize: '10px' }}>▸</span>
                            {group.label}
                            <span style={{ marginLeft: 'auto', fontSize: '10px', color: 'var(--color-text-secondary)', fontWeight: 500 }}>
                              {group.items.length}
                            </span>
                          </div>
                          {group.items.map((sample) => {
                            const globalIndex = filteredSamples.findIndex((s) => s.id === sample.id);
                            const isHighlighted = globalIndex === highlightedIndex;
                            const isSelected = String(sample.id) === String(selectedSampleId);
                            return (
                              <div
                                key={sample.id}
                                onClick={() => handleSelect(sample.id)}
                                onMouseEnter={() => setHighlightedIndex(globalIndex)}
                                style={{
                                  padding: '8px 10px',
                                  cursor: 'pointer',
                                  fontSize: 'var(--font-size-sm)',
                                  background: isHighlighted
                                    ? 'var(--color-accent, #0066cc)'
                                    : isSelected
                                      ? 'var(--color-surface-secondary, #f0f0f0)'
                                      : 'transparent',
                                  color: isHighlighted ? '#fff' : 'inherit',
                                  borderLeft: isSelected ? '3px solid var(--color-accent)' : '3px solid transparent',
                                }}
                              >
                                <div style={{ fontWeight: 500 }}>[{sample.id}]</div>
                                <div style={{ opacity: 0.9, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                  {sample.question.slice(0, 70)}{sample.question.length > 70 ? '...' : ''}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      ))
                    )}
                  </div>
                  <div
                    style={{
                      padding: '6px 10px',
                      fontSize: 'var(--font-size-xs)',
                      color: 'var(--color-text-secondary)',
                      borderTop: '1px solid var(--color-border)',
                      textAlign: 'center',
                    }}
                  >
                    {filteredSamples.length} of {flattenedSamples.length} samples
                  </div>
                </div>
              )}
            </div>
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
                          ? 'Agentic mode includes the controller\'s tool calls, verification and uncertainty signals.'
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
                            <span className="summary-stat-label" data-tooltip="Confidence label of the selected system's final answer (for the agentic loop, set by the verifier).">Selected confidence</span>
                            <span className={`tag ${getConfidenceTagClass(selectedConfidence)}`}>
                              {selectedConfidence}
                            </span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Based on the Verification Agent's checks. 'Verified' requires the review to pass its checks (question matched, evidence grounded, strength appropriate, no contradictions, format correct) and an independent blind reading of the image to agree with the answer. 'Partially_verified' means the review found problems or the blind reading disagrees.">Verification</span>
                            <span className={`tag ${getVerificationTagClass(overallVerification)}`}>
                              {overallVerification.replace(/_/g, ' ')}
                            </span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Self-consistency: the reasoner is re-asked independently (default 2 times) and the re-answers are compared with the final answer. High = all agree; medium = at least half agree; low = fewer than half agree (the case is flagged for review).">Self-consistency</span>
                            <span className={`tag ${getConfidenceTagClass(selfConsistencyLevel)}`}>
                              {selfConsistencyLevel}
                            </span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Evaluates whether the answer is grounded in the visual evidence AND free of contradictions with the image itself. 'High' = both true; 'low' = either false; 'medium' otherwise.">Cross-modal agreement</span>
                            <span className={`tag ${getConfidenceTagClass(crossModalAgreement)}`}>{crossModalAgreement}</span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Set to 'yes' when the verification status is low confidence or unsupported, a high-severity failure category is found (e.g. unsupported claim, visual contradiction, parser failure), or the re-answers are self-inconsistent.">Review recommended</span>
                            <span className={`tag ${reviewRecommended ? 'tag-red' : 'tag-green'}`}>
                              {reviewRecommended ? 'yes' : 'no'}
                            </span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Number of issues the verifier found in the kept (best-verified) answer.">Issues found</span>
                            <span className={`tag ${issueCount === 0 ? 'tag-green' : 'tag-amber'}`}>{issueCount}</span>
                          </div>
                        </div>
                      </section>

                      <section className="summary-section">
                        <div className="summary-section-title">Controller</div>
                        <div className="summary-stat-list">
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="What the agentic controller did: direct (look, answer, check, finish), relook (focused re-examination of a region), reanswer (answer revised after a failed check) or abstained.">Path</span>
                            <span className={`tag ${getControllerPathTagClass(controllerPath)}`}>{controllerPath}</span>
                          </div>
                          <div className="summary-stat-row summary-stat-row-inline">
                            <span className="summary-stat-label" data-tooltip="Number of answers produced, and which one was kept (the best-verified answer).">Answers</span>
                            <span className="summary-stat-value">{answerAttempts ? `${answerAttempts} (kept #${selectedAttempt})` : '—'}</span>
                          </div>
                          <div className="summary-summary-text">
                            {controllerActions.length ? controllerActions.join(' → ') : 'No controller actions recorded for this case.'}
                          </div>
                        </div>
                      </section>
                    </>
                  ) : (
                    <section className="summary-section summary-section-wide">
                      <div className="summary-section-title">Selected Result</div>
                      <div className="summary-stat-list">
                        <div className="summary-stat-row summary-stat-row-inline">
                          <span className="summary-stat-label" data-tooltip="Confidence label of the selected system's final answer (for the agentic loop, set by the verifier).">Selected confidence</span>
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
                      ? 'Workflow Baseline: a fixed sequence of steps without self-correction.'
                      : 'Single-shot Baseline: a direct, single-prompt approach without intermediate decomposition.'}
                    {' '}Switch to Agentic Loop to inspect the controller's steps and verification details.
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
