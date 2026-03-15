import { useState, useEffect, useCallback } from 'react';

import { API_BASE } from '../config';

export function useDataset(initialSplit = 'all') {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeSplit, setActiveSplit] = useState(initialSplit);

  const fetchSamples = useCallback(async (split = activeSplit) => {
    setLoading(true);
    setError(null);

    try {
      const splitParam = split && split !== 'all' ? `&split=${split}` : '';
      const response = await fetch(`${API_BASE}/api/dataset/samples?limit=30${splitParam}`);
      if (!response.ok) throw new Error('Failed to fetch samples');

      const data = await response.json();
      setSamples(data.samples || []);
    } catch (err) {
      setError(err.message);
      setSamples([]);
    } finally {
      setLoading(false);
    }
  }, [activeSplit]);

  const fetchSampleAnalysis = useCallback(async (sampleId, refresh = false) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/dataset/sample/${sampleId}/analysis?refresh=${refresh ? 'true' : 'false'}`
      );
      if (!response.ok) throw new Error('Sample analysis not found');

      const data = await response.json();
      return data;
    } catch {
      return null;
    }
  }, []);

  // Fetch on mount or when activeSplit changes organically
  useEffect(() => {
    fetchSamples();
  }, [fetchSamples]);

  // Expose changeSplit for manual UI triggers
  const changeSplit = useCallback(async (split) => {
    setActiveSplit(split);
    // Explicit return to allow chaining, react will batch the state update
  }, []);

  return {
    samples,
    loading,
    error,
    activeSplit,
    fetchSampleAnalysis,
    changeSplit,
    refetch: fetchSamples,
  };
}
