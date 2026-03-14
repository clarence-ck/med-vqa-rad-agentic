export default function ConfidenceBadge({ level }) {
  const normalized = (level || 'low').toLowerCase();

  const config = {
    high: { label: 'High Confidence', className: 'confidence-high' },
    medium: { label: 'Medium Confidence', className: 'confidence-medium' },
    low: { label: 'Low Confidence', className: 'confidence-low' },
    unverified: { label: 'Not Verified', className: 'confidence-unverified' },
  };

  const { label, className } = config[normalized] || config.low;

  return (
    <span className={`confidence-badge ${className}`}>
      <span className="dot"></span>
      {label}
    </span>
  );
}
