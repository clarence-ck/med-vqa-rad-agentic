import { useCallback, useState } from 'react';

import { API_BASE } from '../config';

export function useEvaluation() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const fetchEvaluation = useCallback(async (split = 'all') => {
    setLoading(true);
    setError(null);
    try {
      const splitParam = split && split !== 'all' ? `?split=${split}` : '';
      const response = await fetch(`${API_BASE}/api/dataset/evaluation${splitParam}`);
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      const payload = await response.json();
      setData(payload);
      return payload;
    } catch (err) {
      const message = err.message || 'Failed to load evaluation data';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    data,
    error,
    fetchEvaluation,
  };
}
