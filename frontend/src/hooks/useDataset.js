import { useState, useEffect, useCallback } from 'react';

import { API_BASE } from '../config';

export function useDataset() {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchSamples = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/dataset/samples?limit=30`);
      if (!response.ok) throw new Error('Failed to fetch samples');

      const data = await response.json();
      setSamples(data.samples || []);
    } catch (err) {
      setError(err.message);
      setSamples([]);
    } finally {
      setLoading(false);
    }
  }, []);

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

  useEffect(() => {
    fetchSamples();
  }, [fetchSamples]);

  return {
    samples,
    loading,
    error,
    fetchSampleAnalysis,
    refetch: fetchSamples,
  };
}
