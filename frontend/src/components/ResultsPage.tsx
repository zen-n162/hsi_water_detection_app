import type { InferenceResponse } from '../lib/api';

type Props = {
  inference: InferenceResponse | null;
  onBackToRoi: () => void;
};

function ResultCard({ title, url }: { title: string; url?: string }) {
  return (
    <div className="result-card">
      <div className="result-card-title">{title}</div>
      {url ? <img src={url} alt={title} /> : <div className="empty-panel">No image.</div>}
    </div>
  );
}

export default function ResultsPage({ inference, onBackToRoi }: Props) {
  return (
    <section className="page-section">
      <div className="page-title-row">
        <h2>Results</h2>
        <button type="button" onClick={onBackToRoi}>
          Back to ROI Selection
        </button>
      </div>

      <div className="results-grid">
        <ResultCard title="Pseudo Color" url={inference?.pseudocolorUrl} />
        <ResultCard title="H2O Detection Overlay" url={inference?.probabilityOverlayUrl} />
        <ResultCard title="Spatial Attention" url={inference?.spatialAttentionOverlayUrl} />
        <ResultCard title="Spectral Attention" url={inference?.spectralAttentionUrl} />
      </div>
    </section>
  );
}
