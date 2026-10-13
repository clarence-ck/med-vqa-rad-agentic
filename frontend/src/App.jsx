import { useState, useCallback } from 'react';
import './index.css';
import CaseHeader from './components/CaseHeader';
import CaseInput from './components/CaseInput';
import ImageViewer from './components/ImageViewer';
import AnswerCard from './components/AnswerCard';
import ExplanationCard from './components/ExplanationCard';
import AgentTrace from './components/AgentTrace';
import VerificationPanel from './components/VerificationPanel';
import ComparisonToggle from './components/ComparisonToggle';
import CaseExplorer from './components/CaseExplorer';
import EvaluationView from './components/EvaluationView';
import FailureModeDrawer from './components/FailureModeDrawer';
import RoutingCard from './components/RoutingCard';
import { useAnalysis } from './hooks/useAnalysis';
import { useDataset } from './hooks/useDataset';
import { useEvaluation } from './hooks/useEvaluation';

function App() {
  const { loading, loadingMessage, result, baselineResult, workflowResult, error, analyze, analyzeBaseline, analyzeWorkflow, reset } = useAnalysis();
  const {
    samples: explorerSamples,
    loading: samplesLoading,
    fetchSampleAnalysis,
  } = useDataset();
  const { loading: evaluationLoading, data: evaluationData, fetchEvaluation } = useEvaluation();
  const [imageSrc, setImageSrc] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [comparisonMode, setComparisonMode] = useState('agentic');
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedExplorerSampleId, setSelectedExplorerSampleId] = useState('');
  const [caseAnalysisLoading, setCaseAnalysisLoading] = useState(false);
  const [caseAnalysisData, setCaseAnalysisData] = useState(null);
  const hasAnyResult = Boolean(result || workflowResult || baselineResult);

  const formatConfidenceMeta = (confidence) => {
    if (!confidence) {
      return { label: 'Not Verified', className: 'tag-gray' };
    }

    const normalized = confidence.toLowerCase();
    if (normalized === 'unknown' || normalized === 'unverified') {
      return { label: 'Not Verified', className: 'tag-gray' };
    }

    if (normalized === 'high') return { label: 'High Confidence', className: 'tag-green' };
    if (normalized === 'medium') return { label: 'Medium Confidence', className: 'tag-amber' };
    return { label: 'Low Confidence', className: 'tag-red' };
  };

  const handleImageSelect = useCallback((file) => {
    setImageFile(file);
    const url = URL.createObjectURL(file);
    setImageSrc(url);
  }, []);

  const handleSubmit = useCallback(async (file, question) => {
    setCurrentQuestion(question);
    reset();

    await Promise.all([
      analyze(file, question),
      analyzeWorkflow(file, question),
      analyzeBaseline(file, question),
    ]);
  }, [analyze, analyzeBaseline, analyzeWorkflow, reset]);

  const handleComparisonToggle = useCallback((mode) => {
    setComparisonMode(mode);
  }, []);

  const handleClear = useCallback(() => {
    setImageSrc(null);
    setImageFile(null);
    setCurrentQuestion('');
    reset();
  }, [reset]);

  const handleLoadCaseAnalysis = useCallback(async (sampleId, refresh = false) => {
    if (!sampleId) return;
    setCaseAnalysisLoading(true);
    const data = await fetchSampleAnalysis(sampleId, refresh);
    setCaseAnalysisData(data);
    setCaseAnalysisLoading(false);
  }, [fetchSampleAnalysis]);

  const handleExplorerSelect = useCallback((sampleId) => {
    setSelectedExplorerSampleId(sampleId);
    if (sampleId) {
      handleLoadCaseAnalysis(sampleId, false);
    } else {
      setCaseAnalysisData(null);
    }
  }, [handleLoadCaseAnalysis]);

  const handleExplorerClear = useCallback(() => {
    setSelectedExplorerSampleId('');
    setCaseAnalysisData(null);
  }, []);

  const handleTabChange = useCallback(async (tab) => {
    setActiveTab(tab);
    if (tab === 'evaluation') {
      await fetchEvaluation();
    }
  }, [fetchEvaluation]);

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-logo">
          <img src="/favicon_transparent.png" alt="MedVQA Logo" className="app-logo-img" />
          <div>
            <div className="app-logo-text">MedVisionQuery</div>
            <div className="app-logo-subtitle">Medical Image QA Assistant</div>
          </div>
        </div>
        <nav className="app-header-nav">
          <button
            className={`nav-tab ${activeTab === 'dashboard' ? 'active' : ''}`}
            id="nav-dashboard"
            onClick={() => handleTabChange('dashboard')}
          >
            Dashboard
          </button>
          <button
            className={`nav-tab ${activeTab === 'explorer' ? 'active' : ''}`}
            id="nav-explorer"
            onClick={() => handleTabChange('explorer')}
          >
            Case Explorer
          </button>
          <button
            className={`nav-tab ${activeTab === 'evaluation' ? 'active' : ''}`}
            id="nav-evaluation"
            onClick={() => handleTabChange('evaluation')}
          >
            Evaluation
          </button>
        </nav>
      </header>

      {activeTab === 'dashboard' && result && (
        <CaseHeader 
          result={result} 
          baselineResult={baselineResult} 
          workflowResult={workflowResult} 
        />
      )}

      {/* Error Display */}
      {error && (
        <div className="error-banner">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {activeTab === 'dashboard' ? (
        <main className="dashboard">
          <aside className="panel">
            <div className="panel-header">
              <h2><span className="icon">📁</span> Case Input</h2>
            </div>
            <CaseInput
              onSubmit={handleSubmit}
              loading={loading}
              onImageSelect={handleImageSelect}
              currentImage={imageFile}
              currentQuestion={currentQuestion}
              onQuestionChange={setCurrentQuestion}
              onClear={handleClear}
              locked={loading || hasAnyResult}
            />
          </aside>

          <section className="panel" style={{ borderRight: '1px solid var(--color-border)' }}>
            <div className="panel-header panel-header-action">
              <h2><span className="icon">🖼️</span> Image Workspace</h2>
              {hasAnyResult && (
                <ComparisonToggle mode={comparisonMode} onToggle={handleComparisonToggle} />
              )}
            </div>
            <ImageViewer imageSrc={imageSrc} result={result} />
          </section>

          <aside className="panel">
            <div className="panel-header">
              <h2><span className="icon">🧠</span> AI Analysis</h2>
            </div>
            <div className="panel-body">
              {!hasAnyResult && !loading && (
                <div className="empty-state">
                  <span className="icon">🔬</span>
                  <div className="empty-state-title">No Analysis Yet</div>
                  <div className="empty-state-text">
                    Upload a medical image and ask a question to begin analysis.
                  </div>
                </div>
              )}

              {loading && (
                <div className="loading-container loading-container-padded">
                  <div className="loading-spinner"></div>
                  <div className="loading-text">{loadingMessage || 'Running analyses...'}</div>
                  <AgentTrace loading={true} />
                </div>
              )}

              {comparisonMode === 'agentic' && result && (
                <>
                  <AnswerCard result={result} />
                  <ExplanationCard result={result} />
                  <AgentTrace agentTrace={result.agent_trace} />
                  <RoutingCard result={result} />
                  <VerificationPanel result={result} />
                  <FailureModeDrawer result={result} />
                </>
              )}

              {comparisonMode === 'baseline' && (
                <>
                  {baselineResult ? (
                    <div className="card animate-fade-in">
                      <div className="card-header">
                        <span className="card-title">⚡ Baseline (Single-shot) Result</span>
                      </div>
                      <div className="card-content">
                        <p className="answer-text">{baselineResult.answer}</p>
                        {baselineResult.explanation && (
                          <p className="explanation-text" style={{ marginTop: 'var(--space-3)' }}>
                            {baselineResult.explanation}
                          </p>
                        )}
                        {baselineResult.confidence && (
                          <div style={{ marginTop: 'var(--space-3)' }}>
                            <span className={`tag ${formatConfidenceMeta(baselineResult.confidence).className}`}>
                              {formatConfidenceMeta(baselineResult.confidence).label}
                            </span>
                          </div>
                        )}
                      </div>
                      <div className="baseline-note">
                        <div className="baseline-note-text">
                          ℹ️ This is a single-shot result without agent decomposition, routing, or verification.
                          Switch to <strong>Agentic Pipeline</strong> to see the full analysis.
                        </div>
                      </div>
                    </div>
                  ) : (
                    !loading && hasAnyResult && (
                      <div className="empty-state">
                        <span className="icon">⚡</span>
                        <div className="empty-state-title">Baseline Result Unavailable</div>
                        <div className="empty-state-text">The single-shot baseline did not produce a result for this case.</div>
                      </div>
                    )
                  )}
                </>
              )}

              {comparisonMode === 'workflow' && (
                <>
                  {workflowResult ? (
                    <div className="card animate-fade-in">
                      <div className="card-header">
                        <span className="card-title">🧭 Workflow Baseline Result</span>
                      </div>
                      <div className="card-content">
                        <p className="answer-text">{workflowResult.answer}</p>
                        {workflowResult.explanation && (
                          <p className="explanation-text" style={{ marginTop: 'var(--space-3)' }}>
                            {workflowResult.explanation}
                          </p>
                        )}
                        {workflowResult.confidence && (
                          <div style={{ marginTop: 'var(--space-3)' }}>
                            <span className={`tag ${formatConfidenceMeta(workflowResult.confidence).className}`}>
                              {formatConfidenceMeta(workflowResult.confidence).label}
                            </span>
                          </div>
                        )}
                      </div>
                      <div className="baseline-note">
                        <div className="baseline-note-text">
                          ℹ️ This is a workflow baseline result with a fixed sequence of analysis steps without agentic self-correction.
                          Switch to <strong>Agentic Pipeline</strong> to see the full analysis with step-by-step reasoning.
                        </div>
                      </div>
                    </div>
                  ) : (
                    !loading && hasAnyResult && (
                      <div className="empty-state">
                        <span className="icon">🧭</span>
                        <div className="empty-state-title">Workflow Result Unavailable</div>
                        <div className="empty-state-text">The workflow baseline did not produce a result for this case.</div>
                      </div>
                    )
                  )}
                </>
              )}
            </div>
          </aside>
        </main>
      ) : activeTab === 'explorer' ? (
        <CaseExplorer
          samples={explorerSamples}
          loading={samplesLoading}
          selectedSampleId={selectedExplorerSampleId}
          onSelectSample={handleExplorerSelect}
          analysisLoading={caseAnalysisLoading}
          analysisData={caseAnalysisData}
          onRefresh={() => handleLoadCaseAnalysis(selectedExplorerSampleId, true)}
          onClearSelection={handleExplorerClear}
        />
      ) : (
        <EvaluationView
          loading={evaluationLoading}
          evaluationData={evaluationData}
          onRefresh={() => fetchEvaluation()}
        />
      )}
    </div>
  );
}

export default App;
