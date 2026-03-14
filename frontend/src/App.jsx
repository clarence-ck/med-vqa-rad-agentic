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
import { useAnalysis } from './hooks/useAnalysis';
import { useDataset } from './hooks/useDataset';
import { useEvaluation } from './hooks/useEvaluation';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function App() {
  const { loading, loadingMessage, result, baselineResult, workflowResult, error, analyze, analyzeBaseline, analyzeWorkflow, reset } = useAnalysis();
  const { samples: dashboardSamples } = useDataset('all');
  const {
    samples: explorerSamples,
    loading: samplesLoading,
    fetchSampleAnalysis,
    activeSplit: explorerSplit,
    changeSplit: changeExplorerSplit,
  } = useDataset('train');
  const { loading: evaluationLoading, data: evaluationData, fetchEvaluation } = useEvaluation();
  const [imageSrc, setImageSrc] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [comparisonMode, setComparisonMode] = useState('agentic');
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedExplorerSampleId, setSelectedExplorerSampleId] = useState('');
  const [caseAnalysisLoading, setCaseAnalysisLoading] = useState(false);
  const [caseAnalysisData, setCaseAnalysisData] = useState(null);
  const [evaluationSplit, setEvaluationSplit] = useState('test');
  const [explorerTestUnlocked, setExplorerTestUnlocked] = useState(false);
  const hasAnyResult = Boolean(result || workflowResult || baselineResult);

  const handleImageSelect = useCallback((file) => {
    setImageFile(file);
    const url = URL.createObjectURL(file);
    setImageSrc(url);
  }, []);

  const handleSampleSelect = useCallback(async (sample) => {
    // If sample has an id >= 0, try fetching from backend
    if (sample.id >= 0) {
      try {
        const response = await fetch(`${API_BASE}/api/dataset/sample/${sample.id}`);
        const data = await response.json();

        if (data.image_base64 && data.image_mime) {
          // Convert base64 to blob/file
          const byteChars = atob(data.image_base64);
          const byteArray = new Uint8Array(byteChars.length);
          for (let i = 0; i < byteChars.length; i++) {
            byteArray[i] = byteChars.charCodeAt(i);
          }
          const blob = new Blob([byteArray], { type: data.image_mime });
          const file = new File([blob], data.image_name || 'sample.png', { type: data.image_mime });

          setImageFile(file);
          setImageSrc(`data:${data.image_mime};base64,${data.image_base64}`);
        }
      } catch (err) {
        console.warn('Could not load sample image:', err);
      }
    }
    setCurrentQuestion(sample.question);
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

  const handleTabChange = useCallback(async (tab) => {
    setActiveTab(tab);
    if (tab === 'evaluation') {
      await fetchEvaluation(evaluationSplit);
    }
  }, [evaluationSplit, fetchEvaluation]);

  const handleExplorerSplitChange = useCallback(async (split) => {
    if (split === 'test' && !explorerTestUnlocked) {
      const confirmed = window.confirm(
        'Test cases should usually be reserved for final evaluation. Open test cases in Case Explorer anyway?'
      );
      if (!confirmed) return;
      setExplorerTestUnlocked(true);
    }
    setSelectedExplorerSampleId('');
    setCaseAnalysisData(null);
    await changeExplorerSplit(split);
  }, [changeExplorerSplit, explorerTestUnlocked]);

  const handleEvaluationSplitChange = useCallback(async (split) => {
    setEvaluationSplit(split);
    if (activeTab === 'evaluation') {
      await fetchEvaluation(split);
    }
  }, [activeTab, fetchEvaluation]);

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-icon">🔬</div>
          <div>
            <div className="app-logo-text">ClariScan</div>
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

      {activeTab === 'dashboard' && result && <CaseHeader result={result} />}

      {/* Error Display */}
      {error && (
        <div style={{
          padding: 'var(--space-3) var(--space-6)',
          background: 'var(--color-error-bg)',
          borderBottom: '1px solid rgba(239, 68, 68, 0.2)',
          color: 'var(--color-error)',
          fontSize: 'var(--font-size-sm)',
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--space-2)'
        }}>
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
              samples={dashboardSamples}
              onSubmit={handleSubmit}
              loading={loading}
              onImageSelect={handleImageSelect}
              onSampleSelect={handleSampleSelect}
              currentImage={imageFile}
              currentQuestion={currentQuestion}
              onQuestionChange={setCurrentQuestion}
            />
          </aside>

          <section className="panel" style={{ borderRight: '1px solid var(--color-border)' }}>
            <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
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
                <div className="loading-container" style={{ padding: 'var(--space-8)' }}>
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
                            <span className={`tag ${
                              baselineResult.confidence === 'high' ? 'tag-green' :
                              baselineResult.confidence === 'medium' ? 'tag-amber' : 'tag-red'
                            }`}>
                              {baselineResult.confidence} confidence
                            </span>
                          </div>
                        )}
                      </div>
                      <div style={{ marginTop: 'var(--space-4)', padding: 'var(--space-3)', background: 'var(--color-bg-input)', borderRadius: 'var(--radius-sm)' }}>
                        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                          ℹ️ This is a single-shot result without agent decomposition, knowledge support, or verification.
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
                            <span className={`tag ${
                              workflowResult.confidence === 'high' ? 'tag-green' :
                              workflowResult.confidence === 'medium' ? 'tag-amber' : 'tag-red'
                            }`}>
                              {workflowResult.confidence} confidence
                            </span>
                          </div>
                        )}
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
          activeSplit={explorerSplit}
          onSplitChange={handleExplorerSplitChange}
          testAccessEnabled={explorerTestUnlocked}
          selectedSampleId={selectedExplorerSampleId}
          onSelectSample={handleExplorerSelect}
          analysisLoading={caseAnalysisLoading}
          analysisData={caseAnalysisData}
          onRefresh={() => handleLoadCaseAnalysis(selectedExplorerSampleId, true)}
        />
      ) : (
        <EvaluationView
          loading={evaluationLoading}
          evaluationData={evaluationData}
          activeSplit={evaluationSplit}
          onSplitChange={handleEvaluationSplitChange}
          onRefresh={() => fetchEvaluation(evaluationSplit)}
        />
      )}
    </div>
  );
}

export default App;
