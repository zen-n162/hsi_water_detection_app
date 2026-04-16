import { useEffect, useMemo, useRef, useState } from 'react';
import type { PreviewResponse } from '../lib/api';

type RoiRect = {
  rowStart: number;
  rowStop: number;
  colStart: number;
  colStop: number;
};

type Props = {
  preview: PreviewResponse | null;
  roi: RoiRect | null;
  setRoi: (r: RoiRect | null) => void;
  onRunInference: () => void;
  busyInference: boolean;
  canRunInference: boolean;
};

export default function RoiPage({ preview, roi, setRoi, onRunInference, busyInference, canRunInference }: Props) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [anchor, setAnchor] = useState<{ x: number; y: number } | null>(null);

  const width = preview?.imageWidth || 0;
  const height = preview?.imageHeight || 0;

  useEffect(() => {
    if (!preview?.grayscalePreviewUrl) {
      setRoi(null);
    }
  }, [preview?.grayscalePreviewUrl, setRoi]);

  const box = useMemo(() => {
    if (!roi || !width || !height) return null;
    return {
      left: `${(roi.colStart / width) * 100}%`,
      top: `${(roi.rowStart / height) * 100}%`,
      width: `${((roi.colStop - roi.colStart) / width) * 100}%`,
      height: `${((roi.rowStop - roi.rowStart) / height) * 100}%`,
    };
  }, [roi, width, height]);

  function toPixelCoords(clientX: number, clientY: number) {
    const img = imgRef.current;
    if (!img || !width || !height) return null;
    const rect = img.getBoundingClientRect();
    const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
    const y = Math.min(Math.max(clientY - rect.top, 0), rect.height);
    return {
      col: Math.round((x / rect.width) * width),
      row: Math.round((y / rect.height) * height),
    };
  }

  function beginDrag(clientX: number, clientY: number) {
    const pt = toPixelCoords(clientX, clientY);
    if (!pt) return;
    setDragging(true);
    setAnchor({ x: pt.col, y: pt.row });
    setRoi({
      colStart: pt.col,
      colStop: Math.min(width, pt.col + 1),
      rowStart: pt.row,
      rowStop: Math.min(height, pt.row + 1),
    });
  }

  function moveDrag(clientX: number, clientY: number) {
    if (!dragging || !anchor) return;
    const pt = toPixelCoords(clientX, clientY);
    if (!pt) return;
    setRoi({
      colStart: Math.min(anchor.x, pt.col),
      colStop: Math.max(anchor.x, pt.col),
      rowStart: Math.min(anchor.y, pt.row),
      rowStop: Math.max(anchor.y, pt.row),
    });
  }

  function endDrag() {
    setDragging(false);
    setAnchor(null);
  }

  return (
    <section className="page-section">
      <div className="page-title-row">
        <h2>ROI Selection</h2>
        <button onClick={onRunInference} disabled={!canRunInference || busyInference} type="button">
          {busyInference ? 'Running...' : 'Run Inference'}
        </button>
      </div>

      <div className="roi-stage">
        {preview?.grayscalePreviewUrl ? (
          <div
            className="roi-canvas"
            onMouseDown={(e) => beginDrag(e.clientX, e.clientY)}
            onMouseMove={(e) => moveDrag(e.clientX, e.clientY)}
            onMouseUp={endDrag}
            onMouseLeave={endDrag}
          >
            <img
              ref={imgRef}
              src={preview.grayscalePreviewUrl}
              alt="ROI preview"
              onError={() => {
                console.error('[ROI preview image load failed]', preview.grayscalePreviewUrl);
              }}
            />
            {box ? <div className="roi-box" style={box} /> : null}
          </div>
        ) : (
          <div className="empty-panel">
            <div>Preview grayscale image will appear here.</div>
            <div style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
              preview url: {preview?.grayscalePreviewUrl || '(empty)'}
            </div>
          </div>
        )}
      </div>

      <div className="roi-meta">
        <div>crop: row {roi?.rowStart ?? '-'}:{roi?.rowStop ?? '-'}, col {roi?.colStart ?? '-'}:{roi?.colStop ?? '-'}</div>
        <div>image size: {width || '-'} × {height || '-'}</div>
      </div>
    </section>
  );
}
