import { useState } from 'react';

export default function ImageViewer({ imageSrc, result }) {
  const [zoomed, setZoomed] = useState(false);

  const modality = result?.image_analysis?.modality;
  const bodyRegion = result?.image_analysis?.body_region;
  const structures = result?.image_analysis?.visible_structures || [];

  return (
    <div className="panel-body" style={{ padding: 0 }}>
      {/* Image Display */}
      <div
        className={`image-viewer ${zoomed ? 'zoomed' : ''}`}
        onClick={() => imageSrc && setZoomed(!zoomed)}
      >
        {imageSrc ? (
          <img src={imageSrc} alt="Medical image" draggable={false} />
        ) : (
          <div className="image-viewer-placeholder">
            <span className="icon">🖼️</span>
            <span>Please Upload An Image</span>
          </div>
        )}

        {imageSrc && (
          <div className="image-controls">
            <button
              className="image-control-btn"
              onClick={(e) => { e.stopPropagation(); setZoomed(!zoomed); }}
            >
              {zoomed ? '🔍−' : '🔍+'}
            </button>
            <button
              className="image-control-btn"
              onClick={(e) => { e.stopPropagation(); setZoomed(false); }}
            >
              Fit
            </button>
          </div>
        )}
      </div>

      {/* Image Tags */}
      {result && (
        <div style={{ padding: 'var(--space-4) var(--space-5)' }}>
          <div className="image-tags animate-fade-in">
            {modality && modality !== 'unknown' && (
              <span className="tag tag-blue">📷 {modality}</span>
            )}
            {bodyRegion && bodyRegion !== 'unknown' && (
              <span className="tag tag-green">🫁 {bodyRegion}</span>
            )}
            {structures.slice(0, 4).map((s, i) => (
              <span key={i} className="tag tag-gray">{s}</span>
            ))}
            {structures.length > 4 && (
              <span className="tag tag-gray">+{structures.length - 4} more</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
