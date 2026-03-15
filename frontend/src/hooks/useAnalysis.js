import { useState, useCallback } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export function useAnalysis() {
  const [agenticLoading, setAgenticLoading] = useState(false);
  const [baselineLoading, setBaselineLoading] = useState(false);
  const [workflowLoading, setWorkflowLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [baselineResult, setBaselineResult] = useState(null);
  const [workflowResult, setWorkflowResult] = useState(null);
  const [error, setError] = useState(null);

  const loading = agenticLoading || baselineLoading || workflowLoading;
  const loadingMessage =
    agenticLoading && baselineLoading && workflowLoading
      ? 'Running all systems...'
      : agenticLoading
        ? 'Running agentic pipeline...'
        : workflowLoading
          ? 'Running workflow baseline...'
          : baselineLoading
            ? 'Running single-shot baseline...'
            : '';

  const analyze = useCallback(async (imageFile, question) => {
    setAgenticLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('question', question);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      return data;
    } catch (err) {
      setError(err.message || 'Analysis failed');
      return null;
    } finally {
      setAgenticLoading(false);
    }
  }, []);

  const analyzeBaseline = useCallback(async (imageFile, question) => {
    setBaselineLoading(true);
    setBaselineResult(null);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('question', question);

    try {
      const response = await fetch(`${API_BASE}/api/analyze/baseline`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setBaselineResult(data);
      return data;
    } catch (err) {
      const fallback = { status: 'error', answer: 'Baseline failed', explanation: err.message, confidence: 'low' };
      setBaselineResult(fallback);
      return fallback;
    } finally {
      setBaselineLoading(false);
    }
  }, []);

  const analyzeWorkflow = useCallback(async (imageFile, question) => {
    setWorkflowLoading(true);
    setWorkflowResult(null);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('question', question);

    try {
      const response = await fetch(`${API_BASE}/api/analyze/workflow`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setWorkflowResult(data);
      return data;
    } catch (err) {
      const fallback = { status: 'error', answer: 'Workflow baseline failed', explanation: err.message, confidence: 'low' };
      setWorkflowResult(fallback);
      return fallback;
    } finally {
      setWorkflowLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setBaselineResult(null);
    setWorkflowResult(null);
    setError(null);
    setAgenticLoading(false);
    setBaselineLoading(false);
    setWorkflowLoading(false);
  }, []);

  return {
    loading,
    loadingMessage,
    agenticLoading,
    baselineLoading,
    workflowLoading,
    result,
    baselineResult,
    workflowResult,
    error,
    analyze,
    analyzeBaseline,
    analyzeWorkflow,
    reset,
  };
}
