import { useState } from 'react';

export default function FailureModeDrawer({ result }) {
  const [open, setOpen] = useState(false);

  if (!result) return null;

  const issues = result.issues_found || [];
  const fallback = result.fallback_recommendation || 'none';
  const confidence = (result.confidence || 'low').toLowerCase();
  const categories = result.failure_categories || [];

  const failureReasons = [];

  categories.forEach((category) => {
    failureReasons.push({
      icon: category.severity === 'high' ? '🚩' : category.severity === 'medium' ? '⚠️' : 'ℹ️',
      text: `${category.label} — ${category.description}`,
    });
  });

  if (confidence === 'low') {
    failureReasons.push({
      icon: '🔍',
      text: 'Low confidence — the available evidence may be insufficient for a reliable answer.',
    });
  }

  if (result.image_analysis?.image_quality === 'poor') {
    failureReasons.push({
      icon: '🖼️',
      text: 'Image quality is poor — this may affect interpretation accuracy.',
    });
  }

  if (result.question_analysis?.question_clarity === 'ambiguous' ||
      result.question_analysis?.question_clarity === 'unclear') {
    failureReasons.push({
      icon: '❓',
      text: 'The question may be ambiguous — consider rephrasing for clearer results.',
    });
  }

  issues.forEach((issue) => {
    if (failureReasons.some((reason) => reason.text === issue)) return;
    failureReasons.push({ icon: '⚠️', text: issue });
  });

  if (fallback === 'add_disclaimer') {
    failureReasons.push({
      icon: '📋',
      text: 'A safety disclaimer has been added — the answer may require expert verification.',
    });
  }

  if (fallback === 'flag_for_review') {
    failureReasons.push({
      icon: '🚩',
      text: 'This case has been flagged for review — results may be unreliable.',
    });
  }

  if (failureReasons.length === 0) {
    failureReasons.push({
      icon: '✅',
      text: 'No significant failure indicators detected for this analysis.',
    });
  }

  return (
    <div>
      <button
        className="failure-drawer-toggle"
        onClick={() => setOpen(!open)}
        id="failure-mode-toggle"
      >
        <span>🔬 Why This Answer May Fail</span>
        <span style={{ transition: 'transform 0.2s', transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}>
          ▾
        </span>
      </button>

      {open && (
        <div className="failure-drawer-content animate-fade-in">
          {failureReasons.map((reason, i) => (
            <div key={i} className="failure-item">
              <span className="icon">{reason.icon}</span>
              <span>{reason.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
